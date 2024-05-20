#AuotModelBert
import os
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

import bert_sentiment_loss, Model
from Model import EarlyStopping
from revised_utils import appply_test_on_x, appply_test_on_OOD, \
    prepare_data_from_pair, load_data_for_run_single,  load_for_Data_imdb

def run4(trsmodel=None, head='None', pooling ='cls',using_head_clsz=True,model_name = 'bert-base-uncased',ft_type='bitfit',
         rdseed=42,train_on_='ori',str2id={'Positive':1,"Negative":0},using_imdb_fordev='imdb1k',specific_optimizer='adamw',warmup_ratio = 0.05,
         epoach=20,batch_size=16,bts_n = 2, stop_batch=12, lr=5e-5,decay=0.1,expected_lenTrTeV = [3414,976,490],max_length=350,
         loss_type='crsloss',weight=0.2,beta=1,temperature=0.5,distance='cosine',early_stopping_thr=15,gap_steps_F1epoch=1,stop_by='acc',
         label_text=[0,1],test_num_ood=None,device_id=1,using_best_valloss_m =True,test_ood= True,
         printlog=True,save_model_fileindex=None):
    '''
      :param using_imdb_fordev: imdb1k: 1k samples as test set. none: original val 488 as test set.
      :param bts_n:int=2, In ‘syncombimdb’, ori-cf consecutive pairs must be batched together.
      :return ood /amazon,yelp,twitter sst-2 imdb
    '''
    ### ori,rev,comb
    # 设置随机种子以确保可重复性
    seed = rdseed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 加载BERT模型和分词器#return_dict=True
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    if trsmodel is None:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2,return_dict=True)
    else:
        if trsmodel =='SupConModelv3':
            model = Model.SupConModelv3(model_name=model_name, num_labels=2, head=head,pooling_strategy=pooling,
                                            using_head_clsz=using_head_clsz,ft_type=ft_type)

        elif trsmodel == 'SupConSBERTv4':
            model = Model.SupConSBERTv4(sbert_hfpath=model_name, num_labels=2,ft_type=ft_type)
        elif trsmodel == 'SupConT5v5':
            model = Model.SupConT5v5(model_name=model_name,num_labels=2)
        
        else:
            model = trsmodel

    # 将模型移至GPU（如果可用）
    model.to(device)
    # 定义优化器和学习率调度器
    num_epochs = epoach
    if specific_optimizer =='adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=decay) # adam： final_loss = loss + wd * all_weights.pow(2).sum() / 2
    elif specific_optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=decay)  # adamw + weight-dacay 更新，w = w- w。grad*lr - lr*wd*w

    criterion = None
    if loss_type == 'crsloss':
        criterion = lambda output,labels: (output.loss, torch.tensor(-1.).to(device),torch.tensor(-1.).to(device))
   
    elif loss_type =='v2supconloss':
        criterion = bert_sentiment_loss.SupConLossv2(lambda_=weight,beta=beta,device=device,temperature=temperature,distance=distance)

   
    # Train Model
    early_stopping = EarlyStopping(patience=early_stopping_thr)
    if save_model_fileindex is None:
        save_best_model = f'{train_on_}_umodel_{int(weight*10)}_{int(temperature*100)}.pth'
    else:
        save_best_model = f'{train_on_}_{save_model_fileindex}_{int(weight * 10)}_{int(temperature * 100)}.pth'
    if os.path.exists(save_best_model):
        # shutil.rmtree(save_best_model)
        os.remove(save_best_model)
        print(f'remove old {save_best_model}')

    test_dataloader=None
    len_train_batch = stop_batch
    if train_on_.startswith('syn'):
        train_dataloader, valid_dataloader, test_dataloader  = \
            prepare_data_from_pair(tokenizer=tokenizer, batch_size=batch_size,bts_n=bts_n,batch_num=stop_batch,using_imdb_fordev=using_imdb_fordev,
                                   train_on_type=train_on_, str2id=str2id, label_text=label_text,max_length=max_length)
      
    elif train_on_ =='imdb' or train_on_ == 'amazon' or train_on_=='yelp' or train_on_=='twitter' or train_on_ =='amazon3':
        train_dataloader, valid_dataloader, test_dataloader =\
            load_for_Data_imdb(data_set=train_on_,base_model=model_name ,max_length=max_length,batch_size=batch_size)
       
    else:
        train_dataloader, valid_dataloader, test_dataloader = \
            load_data_for_run_single(tokenizer=tokenizer, train_on=train_on_, str2id=str2id,using_imdb_fordev=using_imdb_fordev,label_text=label_text,
                                     batch_size=batch_size, batch_num=stop_batch,expected_lenTrTeV=expected_lenTrTeV, printlog=printlog,rdnseed=seed,max_length=max_length)
    if stop_batch is None: len_train_batch = len(train_dataloader)

    ## Used to calculate steps for early stopping; one epoch has batch num steps
    per_gab_limite = None
    step_gap_counter = 0
    if per_gab_limite is None:
        per_gab_limite = len_train_batch // gap_steps_F1epoch
    
    # Define scheduler
    total_steps = len_train_batch * num_epochs

    warmup_steps = int(total_steps*warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    
    ####start to train the model
    train_stop = False
    train_time_s = time.time()
    for epoch in range(num_epochs):
        # stop_train_batch = 0
        print(f'\n ----------{epoch+1}/{num_epochs}-read_dataset--------')

        isFirst_train_batch = True # for printing the first batch labels
        total_loss,total_crsloss,total_suploss = 0,0,0
        train_total,train_correct = 0,0
        
        for batch in train_dataloader:
      
            model.train()
            batch = tuple(t.to(device) for t in batch)
            input_id = batch[0]
            # input_id.requires_grad_(True)
            atten_mask = batch[1]
            labels = batch[2]
            inputs = {'input_ids': input_id,
                      'attention_mask':atten_mask,
                      'labels': labels}
            optimizer.zero_grad()
            outputs = model(**inputs, output_hidden_states=True)
            
          
            loss,crsloss,suploss = criterion(outputs,labels)
            total_crsloss += crsloss.item() #split loss：crsloss
            total_suploss += suploss.item() # split loss：supconloss
            total_loss += loss.item()
            
                
            ## calculate train correction
            pred_label_tmp = torch.argmax(outputs.logits, dim=-1)
            corr_ = (pred_label_tmp==labels).sum().item()
            train_correct += corr_
            train_total += len(labels)


            if isFirst_train_batch and printlog :
                print(f"NO.{epoch+1}/{num_epochs} true label 1st batch:{labels}")
                print(f"NO.{epoch+1}/{num_epochs} pred label 1st batch:{pred_label_tmp}")
                isFirst_train_batch = False
               
            loss.backward() # calculate gradient
            optimizer.step() # gradient update
            scheduler.step() # adjust learning rate
            step_gap_counter += 1

            if step_gap_counter % per_gab_limite ==0:  #print loss
                # record cur lr
                cur_lr = optimizer.param_groups[0]['lr']
                print(f"lr:{cur_lr},scheL{scheduler.get_last_lr()}")
                # calculate train metrics
                average_train_loss = total_loss / per_gab_limite
                average_train_acc = train_correct / train_total
                print(f'Epoch {epoch + 1}/{num_epochs} [{step_gap_counter}/{len(train_dataloader)*(epoch+1)}]: Train loss = {average_train_loss:.4f}, accuracy = {average_train_acc*100:.4f}%')
             

                # evaluate on EVL
                model.eval()
                total_eval_loss,total_eval_crsloss ,total_eval_suploss  = 0,0,0
                correct,total = 0 , 0
               

                for batch in valid_dataloader:
                    
                    batch = tuple(t.to(device) for t in batch)
                    input_id = batch[0]
                    atten_mask = batch[1]
                    labels = batch[2]

                    inputs = {'input_ids': input_id,
                              'attention_mask': atten_mask,
                              'labels': labels}
                    optimizer.zero_grad()
                    # with torch.no_grad():
                    outputs = model(**inputs,output_hidden_states=True)
                
                  
                    loss ,crsloss,suploss= criterion(outputs, labels)
                    total_eval_crsloss += crsloss.item()
                    total_eval_suploss += suploss.item()
                    total_eval_loss += loss.item()
                    
            
                    # accuracy
                    logits =torch.sigmoid(outputs.logits)
                    predicted_labels = torch.argmax(logits,dim=1)
                    corr_ = (predicted_labels == labels).sum().item()
                    correct += corr_
                    total += len(labels)



                average_eval_loss = total_eval_loss / len(valid_dataloader)
                average_eval_acc = correct / total
                print(f'Epoch {epoch+1}/{num_epochs} [{step_gap_counter}/{len(train_dataloader)*(epoch+1)}] Validation loss = {average_eval_loss:.4f}, accuracy = {average_eval_acc*100:.4f}%')
             

                # early stopping
                best_stop_value = average_eval_loss
                if stop_by in ['acc','Acc','accuracy']:
                    best_stop_value = 1 - average_eval_acc
                if best_stop_value < early_stopping.best_val_loss:
                    torch.save(model.state_dict(),save_best_model)
                    early_stopping.stop_epoch= f"{epoch+1}/[{step_gap_counter}/{len(train_dataloader)*(epoch+1)},{per_gab_limite}]"
                    print(f'update {save_best_model}')

                if early_stopping(best_stop_value):
                    train_stop= True
                    print('Early stopping')
                    break
                total_loss, total_crsloss, total_suploss = 0, 0, 0
                train_total, train_correct = 0, 0
               

        if train_stop:
            break

    train_time_sec = time.time() - train_time_s
   
    try:
        ### Use the optimal model parameter， OR it will take the last one.
        if using_best_valloss_m:
            model.load_state_dict(torch.load(save_best_model))
   
    finally:

        model.eval()
        # Evaluate on TEST set
        total_test_loss = 0
        total = 0
        correct = 0
        for batch in test_dataloader:
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids': batch[0],
                      'attention_mask': batch[1],
                      'labels': batch[2]}
            labels= batch[2]
            # with torch.no_grad():
            optimizer.zero_grad()
            outputs = model(**inputs,output_hidden_states=True)
          
            loss,_,_ = criterion(outputs, labels=batch[2])
            # loss = outputs.loss
            total_test_loss += loss.item()

            logits = torch.sigmoid(outputs.logits)
            predicted_labels = (torch.argmax(logits, dim=1) > 0.5).long()
            correct += (predicted_labels == batch[2]).sum().item()
            total += len(batch[2])
        
        average_test_loss = total_test_loss / len(test_dataloader)
        accuracy = correct / total
        print(f'{train_on_} Test loss = {average_test_loss:.4f}')
        print(f'{train_on_} Test Accuracy = {accuracy * 100:.4f}%')
       
        if test_ood:
            amaood, amaood3,yelpood,twitood ,sst2ood=0,0,0,0,0
            orit = appply_test_on_x(model=model,tokenizer=tokenizer, test_type='test_ori',device_id=device_id,max_length=max_length)
            revt = appply_test_on_x(model=model,tokenizer=tokenizer,test_type='test_rev',device_id=device_id,max_length=max_length)
            amaood3 = appply_test_on_OOD(model=model, tokenizers=tokenizer, oodata='amazon3', test_num=test_num_ood, raw_bert=False, device_id=device_id,max_length=max_length)
            yelpood =appply_test_on_OOD(model=model,tokenizers=tokenizer,oodata='yelp',test_num=test_num_ood,raw_bert=False,device_id=device_id,max_length=max_length)
            imdbood = appply_test_on_OOD(model=model,tokenizers=tokenizer,oodata='imdb',test_num=test_num_ood,raw_bert=False,device_id=device_id,max_length=max_length)
            twitood = appply_test_on_OOD(model=model,tokenizers=tokenizer,oodata='twitter',test_num=test_num_ood,device_id=device_id,raw_bert=False,max_length=max_length)
          
            sst2ood = appply_test_on_OOD(model=model, tokenizers=tokenizer, oodata='sst-2', test_num=None,
                                         device_id=device_id, raw_bert=False, max_length=max_length)
           
            return save_best_model, (train_time_sec, accuracy,orit, revt, amaood3,yelpood,twitood,sst2ood,imdbood)
           
        else:
            return  save_best_model,(train_time_sec,accuracy)


