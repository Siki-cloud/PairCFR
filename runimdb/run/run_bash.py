from Re_Update_few_shot_exp import run4

def run_fewshotFullexp(loss = 'crsloss', train_on = 'comb17',bts=32, lr = 5e-5,decay = 0.1,opt='adamw',add_data =None,gap=10,stopbatch=None,ft_type='bitfit',stop_by='acc',epoach=20):
    '''
    :param: specific_pooling = 'cls_pooler' #v3:cls, first_last_avg,dct,last_avg,cls_pooler
    :param: specific_head = None  # None ,linear, mlp ,pca
    :param: specific_head_clsz = True #true -> classifier(head(pooling_feature)),false->classifier(output_pooler)
    '''
    specific_pooling, specific_head, specific_head_clsz = 'None','None','None'
    
    ###### bert-based
    # specific_model ='SupConModelv3'
    # base_model, specific_pooling, specific_head, specific_head_clsz = 'roberta-base', 'cls', None, False
    # base_model,specific_pooling,specific_head,specific_head_clsz = 'bert-base-uncased','cls',None,False

    # base_model, specific_pooling, specific_head, specific_head_clsz ='distilbert-base-uncased','cls','linear',True
    # base_model, specific_pooling, specific_head, specific_head_clsz = 'albert-base-v2','cls_pooler',None,True
    
    ##### t5 
    # specific_model = 'SupConT5v5'
    # base_model = 't5-base'
    ### sbert
    specific_model = 'SupConSBERTv4'
    base_model = 'sentence-transformers/multi-qa-distilbert-cos-v1'
    
    using_imdb_fordev = None  # imdb1k, None,imdb1.7k,imdbfull
    if stopbatch<0:
        stopbatch = None

    print(base_model)   
    if add_data is None:
        expected_lenTrTeV=None
    else:
        expected_lenTrTeV = [3414, 976, 490] #Increase ori, double the size of the training set from 1707 to 3414, with the test set at 976 and the dev set at 490 proportionally.
    batch_num = stopbatch # None # Limit the number of training samples according to the batch size. If set to None, train on the full dataset

    bts_n = 2 # In the case of syscomb with a fixed value of 2, it indicates that in pair files, consecutive pairs of data (ori, cf) are bundled together.
    res = {} # Store the result of ind -ood
    for seed in [2]:  
        
        for w in [0.9]:  ## roberta-> w=0.9，t=0.07
            beta = 1 - w
            for t in [0.07]:  
                if loss.endswith('conloss') is False:
                    w,t = 1,1
                print(f"bts:{bts},stop_batch:{batch_num}")
                save_best_model, restmp = run4(trsmodel=specific_model,head=specific_head,pooling=specific_pooling,using_head_clsz=specific_head_clsz,
                                               model_name=base_model,ft_type=ft_type,rdseed=seed, train_on_=train_on,
                                               using_imdb_fordev= using_imdb_fordev, 
                                               decay=decay, specific_optimizer=opt, warmup_ratio=0.05,
                                               expected_lenTrTeV= expected_lenTrTeV,  max_length=350,
                                               epoach=epoach, lr=lr, batch_size=bts, bts_n=bts_n, stop_batch=batch_num,gap_steps_F1epoch=gap,
                                               loss_type=loss, weight=w, beta=beta,temperature=t,
                                               early_stopping_thr=5,stop_by=stop_by,
                                               device_id=0, distance='cosine',
                                               using_best_valloss_m=True,
                                               test_num_ood=100, save_model_fileindex='run5x')
                print(f"ood10kres (0,{train_on} {loss},{w},{t},{bts},{seed}):{restmp}")
                if train_on=='ori' and expected_lenTrTeV is None:
                    train_on_ = 'ori17'
                else:
                    train_on_ = train_on
                res[(0,base_model,train_on_,loss,w,beta,t,bts,lr,seed,decay,opt,batch_num,ft_type,stop_by)] = restmp

               
            # t 结束
            for k, v in res.items():
                print(f"{k}:{v}")
    return res

def run_for_fewshot_run5_full():
    ft_typ='full'
    stop_by= 'loss'
    epoach=2
    res_t = {}

    for loss in ['v2supconloss']:#  crsloss  v2supconloss 
        for train_on,add_data in [('syncomb','defalut')]: #('syncomb34','defalut') ('ori','defalut'),('comb','defalut')
           for decay,opt in [(0.1,'adamw')]: #(0.1,'adamw')(0,'adam')
                for lr,gab_ in [(3e-6,1)]:## switch  to the appropriate lr
                    for bts,stop_batch in [(16,-1)]: #, ,(4,4), -1 is full dataset fot training
                        print(lr)
                        res = run_fewshotFullexp(bts=bts, stopbatch=stop_batch,
                                                 loss=loss,
                                                 train_on=train_on,
                                                 lr=lr, gap=gab_,
                                                 add_data=add_data,
                                                 decay=decay, opt=opt,
                                                 ft_type=ft_typ,
                                                 stop_by=stop_by,
                                                 epoach=epoach)
                        res_t.update(res)
        print("for few shot")
        for k,v in res_t.items():
            print(f"{k}:{v}")
    return res_t

if __name__ == '__main__':
    res1 = run_for_fewshot_run5_full()
   
