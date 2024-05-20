from bert_snli import snli

def go_run_snli(lr=5e-5,batch_size =30,stop_batch_num=None,loss_type = 'v2supconloss',epoch=10):

    specific_pooling, specific_head, specific_head_clsz = 'None','None','None'
    ###### bert 系列
    specific_model ='SupConModelv3'
    # base_model, specific_pooling, specific_head, specific_head_clsz = 'roberta-base', 'cls', None, False
    # base_model,specific_pooling,specific_head,specific_head_clsz = 'bert-base-uncased','cls',None,False
    # base_model, specific_pooling, specific_head, specific_head_clsz = 'albert-base-v2','cls_pooler',None,True
    # ft_type='bitfit' # 'bitfit','full'
    ##### t5 系列
    # specific_model = 'SupConT5v5'
    # base_model = 't5-base'
    
    ### sentence bert
    specific_model = 'SupConSBERTv4'
    base_model = 'sentence-transformers/multi-qa-distilbert-cos-v1'
 
    
    decay,opt = 0.1,'adamw'
    # decay, opt = 0, 'adam'
    earlystop_epoch= 3
    max_lenght = 64
    ##In the file, when there are consecutive 5 entries of 1-4, they must be in the same batch for ori:cfs and syn-.
    continue_n_file = 5 #ori_rh/rp 3, rp_rh 4 
    gap_num,stop_by =1,'loss'
   
    global_res = {}  
    #('original6.6',5000),('original8.6',7000) ('original13.6',12000),('original16.6',15000),('original19.6',18000),('original22.6',21000)
    ## ('synori_rh_rp83',-1),('original8.3',6700),('ori_rh_rp83',-1) 
    for train_on,add_n in [ ('synori_rh_rp83',-1)]: # ori <= 550k 
        for seed in [71]: #13,71,89,211 ,52,187,19 ,388,
            for w,t in [(0.7,0.9)]:
                for rnd_gc in [-1]:
                     if not loss_type.endswith('conloss'):
                         w,t=1,1
                     
                     res = snli(rdseed=seed,trsmodel=specific_model, base_model=base_model,ft_type='full',linearlayer=1,
                                head=specific_head,pooling=specific_pooling,using_head_clsz=specific_head_clsz,except_1=True,
                                 decay=decay, lr=lr, batch_size=batch_size, epochs=epoch,warmup_ratio=0.1,rnd_gc=rnd_gc,
                                train_type=train_on,add_ori_train_num=add_n, max_length=max_lenght, early_stopping_thr=earlystop_epoch,
                                loss_type=loss_type,weight=w,temperature=t,continue_comb_n_infile=continue_n_file,
                                gap=gap_num,stop_by=stop_by,pth_fix='nli2',stop_batch_num=stop_batch_num, record_log=False,
                                device_id = 0)
                     # save
                     global_res[(base_model,train_on,rnd_gc,lr,batch_size,stop_batch_num,decay,opt,loss_type,w,t,seed,stop_by)] = res
                     print(f"Train on {train_on}:{res}")
                # t END print
                for k,v in global_res.items():
                    print(f"{k}:{v}")
        # seed END  print
        print('seed')
        for k,v in global_res.items():
            print(f"{k}:{v}")
    print("wt")
    for k,v in global_res.items():
        print(f"{k}:{v}")
    return global_res

def runsnli_for_fullwt():
    global_res ={}
    # batchsize = 30
    lr=3e-5
    #5,16
    for batch_size,stop_num,epoch in [(30,-1,2)]: # few shot learning :(5,10,20),(5,20,20),(10,50,20),(10,100,20),(20,200,20)
        for loss in ['v2supconloss']: #'crsloss'
            tmp=go_run_snli(lr=lr,batch_size=batch_size,stop_batch_num=stop_num,loss_type=loss,epoch=epoch)
            global_res.update(tmp)
    print('final')
    for k,v in global_res.items():
        print(f"{k}:{v}")
    return global_res
if __name__ == '__main__':
    runsnli_for_fullwt()