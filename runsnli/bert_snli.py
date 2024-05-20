import os
import random

import numpy as np
from transformers import  AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
import torch
import time
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
import pandas as pd
import csv
import  sys
# sys.path.append(r'../..')
import Model
from data_utils import load_data_for_trainsingle, apply_snliood_test, apply_snli_indomain, load_data_for_train_comb,load_data_for_train_full
import bert_sentiment_loss

cls_gradients = 0
def snli(rdseed=42,trsmodel='SupConModelv3', base_model ='bert-base-uncased',linearlayer=1,
         head='None', pooling ='cls',using_head_clsz=True,ft_type='full',warmup_ratio=0.05,
         decay = 0.1,opt='adamw',lr=1e-5,batch_size=16,epochs = 20,rnd_gc = None,stop_batch_num =  None,
         device_id=0,train_type='original', add_ori_train_num =-1, continue_comb_n_infile=5,max_length=50,loss_type='celoss',except_1 = True,
         early_stopping_thr=5,gap=10,stop_by='valacc',weight=1,beta=None,temperature=1,pth_fix=None,record_log=False,log_file='log.txt'):
    '''
    rnd_gc :  number of CFEs used for training, None -> 4; option-> 1 2 3
    '''

    
    # Set random seed for reproducibility
    seed = rdseed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 初始化tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    
    if trsmodel is None:
        model = AutoModelForSequenceClassification.from_pretrained(base_model,num_labels=3,return_dict=True)
    else:# 3类分类任务
        print(f"{trsmodel} :{base_model}")
        if trsmodel =='SupConModelv3':
            model = Model.SupConModelv3(model_name=base_model, num_labels=3, head=head,pooling_strategy=pooling,
                                            using_head_clsz=using_head_clsz,ft_type=ft_type,linearlayer=linearlayer)
        elif trsmodel == 'SupConSBERTv4':
            model = Model.SupConSBERTv4(sbert_hfpath=base_model, num_labels=3,ft_type=ft_type,linearlayer=linearlayer)
        elif trsmodel == 'SupConT5v5':
            model = Model.SupConT5v5(model_name=base_model,num_labels=3)
       
        else:
            model = trsmodel
    model.to(device)
    # Loading training data
    if train_type.startswith('syn'):
        train_loader, dev_loader, test_loader = load_data_for_train_comb(train_type=train_type,tokenizer=tokenizer,max_length=max_length,bts_size=batch_size,bts_n=continue_comb_n_infile,rnd_gc=rnd_gc,stop_batch_num=stop_batch_num)
    elif train_type.startswith('full'):
        train_loader, dev_loader, test_loader = load_data_for_train_full(train_type=train_type,tokenizer=tokenizer,max_length=max_length,bts=batch_size)
    else:
        train_loader,dev_loader,test_loader = load_data_for_trainsingle(train_type=train_type,tokenizer=tokenizer,max_length=max_length,bts=batch_size,add_data_n=add_ori_train_num,stop_batch_num=stop_batch_num)

    
    if opt == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=decay)
    elif opt =='adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    if beta is None:
        beta = 1 - weight
    
    ## loss
    if loss_type =='crsloss':
        criterion = lambda output,labels: (outputs.loss, torch.tensor(-1).to(device),torch.tensor(-1).to(device))
    
    elif loss_type =='v2supconloss':
        criterion = bert_sentiment_loss.SupConLossv2(lambda_=weight,beta=beta,device=device,temperature=temperature,except_1=except_1)
   
   
    early_stopping = Model.EarlyStopping(patience=early_stopping_thr)
    if pth_fix is None:
        save_best_model = f'{train_type}_snli_{base_model.split("-")[0]}{int(weight*10)}_{int(temperature*100)}.pth'
    else:
        save_best_model = f'{train_type}_snli_{pth_fix}_{base_model.split("-")[0]}{int(weight*10)}_{int(temperature*100)}.pth'
    
    if os.path.exists(save_best_model):
        os.remove(save_best_model)
        print(f'remove old {save_best_model}')
    # Zaoting
    gap_step_judge_stop = len(train_loader)
    if gap is not None and gap >0:
        gap_step_judge_stop  = len(train_loader) // gap
    step_counter = 0
    # define scheduler
    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps*warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
   
    # record changes of VAL loss
    text_for_log = []
    headline=[f"eps/{train_type},{loss_type},w={weight},t={temperature},cfs={rnd_gc},seed={seed}","val loss","ce","scl","acc","f1","best_epoch"]
    text_for_log.append(headline)

    train_stop = False
    train_start = time.time()
    # Train iterative
    for epoch in range(epochs):
        print(f"**{epoch+1}/{epochs}**\ntrain b:{len(train_loader)}, val b:{len(dev_loader)}, test b:{len(test_loader)}")

        train_total = 0
        train_correct = 0
        train_loss = 0
        train_suploss = 0
        train_celoss  =0 
        train_f1 = 0
        ifFirst=True
        for batch in train_loader:
            step_counter  = step_counter +1
          
            model.train()
            optimizer.zero_grad()
            batch = tuple(t.to(device) for t in batch)
            # print(batch)
            input_id ,atten_mask, labels = batch[0], batch[1], batch[2]
            inputs = {'input_ids': input_id,'attention_mask': atten_mask,'labels': labels}
            outputs = model(**inputs,output_hidden_states=True)
           
            loss,celoss,supconloss = criterion(outputs,labels)
            train_celoss += celoss.item()
            train_suploss += supconloss.item()
            train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            scheduler.step() 
            train_pred = torch.argmax(outputs.logits,dim=1)
            train_correct += (train_pred==labels).sum().item()
            train_total += len(labels)
            train_f1 += f1_score(labels.cpu().numpy(), train_pred.cpu().numpy(), average='weighted')
            if ifFirst:
                print(f"first_batch true:{labels}")
                print(f"first_batch pred:{train_pred}")
                ifFirst = False
         
            if step_counter % gap_step_judge_stop == 0:
                # record cur lr
                cur_lr = optimizer.param_groups[0]['lr']
                print(f"lr:{cur_lr},scheL{scheduler.get_last_lr()}")
                # print(f"lr:{cur_lr}")
                # calculate train metrics
                avg_train_loss = train_loss / len(train_loader)
                avg_train_celoss = train_celoss/len(train_loader)
                avg_train_supconloss = train_suploss / len(train_loader)
                avg_train_acc = train_correct / train_total
                train_f1 = train_f1 / len(train_loader)
                print(f"Epoch {epoch+1}/{epochs} [{step_counter}/{len(train_loader)*(epoch+1)}], Train Loss:{avg_train_loss:.5f}/ ce: {avg_train_celoss:.2f}/ scl: {avg_train_supconloss:.2f}, Acc ：{avg_train_acc:.5f}, F1:{train_f1:.5f}")
                # Evaluate on EVL dataset
                model.eval()
                total_eval_loss = 0
                total_eval_celoss = 0
                total_eval_supconloss = 0
                total_eval_correct = 0
                total_eval = 0
                total_eval_f1 = 0
                for batch in dev_loader:
                   
                    batch = tuple(t.to(device) for t in batch)
                    input_id, atten_mask, labels = batch[0], batch[1], batch[2]
                    inputs = {'input_ids': input_id, 'attention_mask': atten_mask, 'labels': labels}
                    # with torch.no_grad():
                    optimizer.zero_grad()
                    outputs = model(**inputs, output_hidden_states=True)
                    
                    
                    loss,celoss,supconloss = criterion(outputs,labels)
                    total_eval_celoss += celoss.item()
                    total_eval_supconloss += supconloss.item()
                    total_eval_loss += loss.item()
                    
                    eval_pred = torch.argmax(outputs.logits,dim=1)
                    total_eval_correct += (eval_pred == labels).sum().item()
                    total_eval += len(labels)
                    total_eval_f1 += f1_score(labels.cpu().numpy(), eval_pred.cpu().numpy(), average='weighted')
                avg_val_loss = total_eval_loss / len(dev_loader)
                avg_val_celoss = total_eval_celoss / len(dev_loader)
                avg_val_scloss = total_eval_supconloss / len(dev_loader)
                avg_val_acc = total_eval_correct/ total_eval
                avg_val_f1 = total_eval_f1 / len(dev_loader)
                print(f'Epoch {epoch + 1}/{epochs} [{step_counter}/{len(train_loader)*(epoch+1)}]- Validation Loss: {avg_val_loss:.5f}/ ce:{avg_val_celoss:.2f}/ scl:{avg_val_scloss:.2f}, Acc: {avg_val_acc:.5f}, F1:{avg_val_f1:.5f}')
                text_for_log.append((f"{epoch + 1}/{epochs}",f"{avg_val_loss:.5f}",f"{avg_val_celoss:.2f}",f"{avg_val_scloss:.2f}",f"{avg_val_acc:.5f}",f"{avg_val_f1:.5f}",f"{early_stopping.stop_epoch}"))
               
                # early stopping
                stop_by_value = avg_val_loss
                if stop_by.endswith('acc') or stop_by.startswith('acc'):
                    stop_by_value = 1 - avg_val_acc

                if stop_by_value < early_stopping.best_val_loss:
                    print(f"{stop_by_value} < {early_stopping.best_val_loss}")
                    torch.save(model.state_dict(), save_best_model)
                    early_stopping.stop_epoch = epoch + 1
                    print(f'update {loss_type}, {save_best_model}')

                if early_stopping(stop_by_value):
                    print(f'Early stopping at {early_stopping.stop_epoch}')
                    train_stop = True
                    break
                train_total = 0
                train_correct = 0
                train_loss = 0
        if train_stop:
                break
        print('--------------------')
    
    train_time = time.time()- train_start
    if record_log: # Record the point of highest val acc during training.
        with open(log_file,'a',newline='',encoding='utf-8') as file:
            tsvwriter = csv.writer(file,delimiter='\t')
            for i in text_for_log:
                tsvwriter.writerow(i)
            file.close()
        
    # Perform final evaluation on the test set
    model.eval()
    predictions = []
    true_labels = []
    test_f1 = 0
    for batch in test_loader:
        batch = tuple(t.to(device) for t in batch)
        input_id, atten_mask, labels = batch[0], batch[1], batch[2]
        inputs = {'input_ids': input_id, 'attention_mask': atten_mask, 'labels': labels}
        # with torch.no_grad():
        optimizer.zero_grad()
        outputs = model(**inputs, output_hidden_states=True)
        logits = outputs.logits
        pred_label  = logits.argmax(dim=1)
        predictions.extend(pred_label.tolist())
        true_labels.extend(labels.tolist())
        test_f1 += f1_score(labels.cpu().numpy(), pred_label.cpu().numpy(), average='weighted')
    # calculate Metric
    correct = sum(p == t for p, t in zip(predictions, true_labels))
    accuracy = correct / len(true_labels)
    test_f1 = test_f1/ len(test_loader)
    print(f'Test Accuracy: {accuracy}, F1-score: {test_f1}')

    model.load_state_dict(torch.load(save_best_model))
    ## indomain & challenge
    snli_in,mmli_mood,mmli_mmood,negation_ood,spellingerror_ood,wordsoverlap_ood=0,0,0,0,0,0
    ori_in = apply_snli_indomain(test_type='ori',model=model,tokenizer=tokenizer, max_length=max_length,batch_size=batch_size,device_id=device_id)
    rh_challeng= apply_snli_indomain(test_type='rh',model=model,tokenizer=tokenizer, max_length=max_length,batch_size=batch_size,device_id=device_id)
    rp_challeng=apply_snli_indomain(test_type='rp',model=model,tokenizer=tokenizer, max_length=max_length,batch_size=batch_size,device_id=device_id)
    snli_in = apply_snliood_test(ood_name='snli', model=model, tokenizer=tokenizer, max_length=max_length,batch_size=batch_size, device_id=device_id, test_num=None)

    ## ood test
    mmli_mmood = apply_snliood_test(ood_name='mnli_mm',model=model,tokenizer=tokenizer,max_length=max_length,batch_size=batch_size,device_id=device_id,test_num=None)
    mmli_mood = apply_snliood_test(ood_name='mnli_m',model=model,tokenizer=tokenizer,max_length=max_length,batch_size=batch_size,device_id=device_id,test_num=None)
    negation_ood = apply_snliood_test(ood_name='negation', model=model, tokenizer=tokenizer, max_length=max_length,batch_size=batch_size, device_id=device_id, test_num=None)
    spellingerror_ood = apply_snliood_test(ood_name='spelling_error', model=model, tokenizer=tokenizer, max_length=max_length,batch_size=batch_size, device_id=device_id, test_num=None)
    wordsoverlap_ood = apply_snliood_test(ood_name='word_overlap', model=model, tokenizer=tokenizer, max_length=max_length,batch_size=batch_size, device_id=device_id, test_num=None)

    return (train_time,ori_in,rh_challeng,rp_challeng,snli_in,mmli_mood,mmli_mmood,negation_ood,spellingerror_ood,wordsoverlap_ood)
