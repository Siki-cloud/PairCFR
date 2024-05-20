
import csv
import random
import datasets
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AutoModelForSequenceClassification, \
    AutoTokenizer 
from torch.utils.data import DataLoader, TensorDataset, Dataset, Sampler
from datasets import load_dataset




def add_data(oridata,expected_len,randomseed=22,mask_index=None,datatype='train',funtype='train',sampletype='random'):
    '''

    :param oridata: list[tuple(2)], tuple(2):(text,0/1)
    :param expected_len: int
    :return: list[tuple(2)]
    '''
    oridata_len = len(oridata)
    add_len = int( expected_len - oridata_len)
    data = datasets.load_dataset('imdb',split= datatype)
    random.seed(randomseed)
    if mask_index is not None and sampletype=='random': # train,val
        candi_index =[x for x in range(len(data)) if x not in mask_index]
        target_index = random.sample(candi_index, add_len)
        mask_index.extend(target_index)
        # print(f'random:{funtype}:{target_index}')
    elif sampletype == 'random' and funtype == 'test': # test
        target_index =random.sample(range(len(data)),add_len)
        # print(f"random:{funtype}:{target_index}")
    elif sampletype=='unifom' and funtype =='val':  # val
        # dev unifom 获取dataset,从后面取
        target_index =  [-x-1 for x in  range(add_len)]
        print(f'unifom:{funtype}:{target_index}')
    else:# train，test unifom 获取dataset，从前面qu

        target_index = range(add_len)
        print(f"unifom:{funtype}:{target_index}")

    sample_data= [data[x] for x in target_index]
    for x in sample_data:
        oridata.append((x['text'],x['label']))

def load_data_for_run_single(tokenizer,data=None,train_on='comb',str2id={'Postive':1,"Negative":0},batch_size=16,expected_lenTrTeV=[3414,976,490],printlog=True,using_imdb_fordev=None,rdnseed=42,max_length=350,batch_num=None,label_text=[0,1]):
    if printlog:
        print('load_data_for_run_single')
    # load dataset
    if data is None and train_on is not None:
        if batch_num is not None:
            batch_num=batch_num * batch_size  # 读取的条数
        train_file, test_file, dev_file = get_train_type(train_on,using_imdb_fordev=using_imdb_fordev)
        train_data = preprocess_data(train_file, str2id,batch_num=batch_num,label_text=label_text)
        test_data = preprocess_data(test_file, str2id, batch_num=None,label_text=label_text)
        val_data = preprocess_data(dev_file, str2id, batch_num=None,label_text=label_text)
    else:
        train_data = data['train']
        test_data = data['test']
        val_data = data['val']
    if train_on == 'ori' and expected_lenTrTeV is not None:
        mask = [0]
        sedd= rdnseed
        if printlog:print(f"add ori data seed :{sedd}")
        add_data(train_data, datatype='train',funtype='train',randomseed=sedd, mask_index=mask, expected_len=expected_lenTrTeV[0])
        add_data(test_data, datatype='test',funtype='test',randomseed=sedd, expected_len=expected_lenTrTeV[1])
        if using_imdb_fordev is None:
            add_data(val_data, datatype='train',funtype='val', randomseed=sedd,mask_index=mask, expected_len=expected_lenTrTeV[2])
        if printlog:print(f"fixed train/test/val data len:{len(train_data)}/{len(test_data)}/{len(val_data)}")
    # max_leng=350
    train_input_ids, train_attention_masks, train_labels = tokenize_data(tokenizer, train_data,max_leng=max_length)
    valid_input_ids, valid_attention_masks, valid_labels = tokenize_data(tokenizer, val_data,max_leng=max_length)
    test_input_ids, test_attention_masks, test_labels = tokenize_data(tokenizer, test_data,max_leng=max_length)
    # 创建数据加载器
    # batch_size = 16
    train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    valid_dataset = TensorDataset(valid_input_ids, valid_attention_masks, valid_labels)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    if printlog: print('------------------')
    return train_dataloader,valid_dataloader,test_dataloader

def prepare_data_from_pair(tokenizer=None,batch_size=16,batch_num=None,
    train_on_type='comb',str2id={'Positive':1,'Negative':0},label_text=[0,1],
    using_imdb_fordev=None,max_length=350,bts_n = 2):

    print(batch_size,batch_num)
    #获取文件路径
    train_pair_file,test_pair_file,dev_pair_file = get_train_type(train_on_=train_on_type,using_imdb_fordev=using_imdb_fordev)
    n = bts_n ## 每隔batch 中连续2条数据捆绑在一起。 ori-cf

    #读文件
    train_data_size = None
    if batch_num is not None and batch_num> 0: train_data_size = batch_size * batch_num
    # 自定义数据类 + sample 类
    train_dataset = IMDbDataset(train_pair_file,tokenizer,max_length,n=bts_n, label_map = str2id,train_data_size=train_data_size)
    train_custom_sample = CustomSampler(train_dataset,batch_size,n=n)

    test_dataset = IMDbDataset(test_pair_file,tokenizer=tokenizer, label_map = str2id, max_length=max_length)
    test_custom_sample = CustomSampler(test_dataset,batch_size,n)

    dev_dataset = IMDbDataset(dev_pair_file,tokenizer,max_length,n=bts_n,label_map=str2id)
    dev_custom_sample = CustomSampler(dev_dataset,batch_size,n)
    
    print(f"\nload dataset:\ntrain len:{len(train_dataset)}, dev len:{len(dev_dataset)}, test len:{len(test_dataset)}")
    # 创建 DataLoader，并使用自定义的 Sampler
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_custom_sample)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, sampler=dev_custom_sample)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_custom_sample)

    return train_loader, dev_loader, test_loader


class IMDbDataset(Dataset):

    def __init__(self, data_path, tokenizer, max_length,label_map,n=5,train_data_size=None):
        self.data = pd.read_csv(data_path, sep='\t', header=0)# 跳过第一行，假设第一行是headline
        self.n = n
        self.label_map = label_map
        # 随机提取指定条数的数据，且控制每2个连续行 同时存在/同时剔除
        if train_data_size is not None and train_data_size > 0 :
            exist_groups_num = train_data_size // n
            indexes = list(range(len(self.data)))
            indexe_groups =  [indexes[i:i + self.n] for i in range(0, len(self.data), self.n)]
           
            groups_indexes = list(range(len(indexe_groups)))
            drop_down_groups_num = len(indexe_groups) - exist_groups_num
            drop_group_index = random.sample(groups_indexes,drop_down_groups_num)
            drop_indexes_row = []
            for i in drop_group_index:
                sub_groups = indexe_groups[i]
                drop_indexes_row.extend(sub_groups)
            newdata = self.data.drop(index=drop_indexes_row).reset_index(drop=True)        
            self.data =  newdata.copy()
            print(f'sampled new data {len(newdata)}')
        print(f'IMDbDataset:{len(self.data)}')
        
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Your implementation here to get item based on index idx
        # idx 是单个单个样本的索引
        data_batch = self.data.iloc[idx]
        ## 查看 cf和 ori 是否同组
        # print(f"data_batch:{data_batch}")
        # print(f"idx:{idx}")
     
        inputs = self.tokenizer(
            data_batch['Text'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        # label_map = {'Positive':1,'Negative':0}
        label_ = self.label_map[data_batch['Sentiment']]
        labels = torch.tensor(label_) # 假设数据集中有'label'列，包含类别标签
        # Retrieve tensors with desired shapes
        input_ids = inputs['input_ids'].squeeze(0)  # Remove the batch dimension if it exists
        attention_mask = inputs['attention_mask'].squeeze(0)  # Remove the batch dimension if it exists
        # return {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask'],
        #         'labels': labels}
        return input_ids,attention_mask,labels

    def shuffle_dataset(self):
        grouped_data = [self.data.iloc[i:i + self.n] for i in range(0, len(self.data), self.n)]
        shuffled_groups = grouped_data.copy()
        random.shuffle(shuffled_groups)
        self.data = pd.concat(shuffled_groups).reset_index(drop=True)

class CustomSampler(Sampler):
    def __init__(self, data_source, batch_size, n):
        self.data_source = data_source
        self.batch_size = batch_size
        self.n = n
  
        self.indices = list(range(len(data_source)))
        self.num_batches = len(self.indices) // self.batch_size

    def __iter__(self):
        # 分小组，n个一组，连续的n条，组内打乱
        indices = [self.indices[i:i + self.n] for i in range(0, len(self.indices), self.n)]
        # random.shuffle(indices) # 如果需要组内打乱，则反注释该行代码
        indices = [item for sublist in indices for item in sublist]
        # 分大组，bts一组，共bts个元素一组 打乱
        batches = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
        random.shuffle(batches)
        indices = [item for sublist in batches for item in sublist]

        return iter(indices[:self.num_batches * self.batch_size])

    def __len__(self):
        return self.num_batches * self.batch_size

def get_train_type(train_on_='comb',using_imdb_fordev=None):
    # 获取训练数据集
    if train_on_=='comb' or train_on_=='comb34':
        train = r'../data/combined34/train_paired.tsv'
        test = r'../data/combined34/test_paired.tsv'
        val = r'../data/combined34/dev_paired.tsv'
    elif train_on_=='ori':
        train = '../data/original17/train.tsv'
        test = '../data/original17/test.tsv'
        val = '../data/original17/dev.tsv'
    elif train_on_=='rev':
        train = '../data/revised17/train.tsv'
        test = '../data/revised17/test.tsv'
        val = '../data/revised17/dev.tsv'
    elif train_on_ =='syncomb34' or train_on_=='syncomb':
        train = r'../data/combined34/train_paired.tsv'
        test = r'../data/combined34/test_paired.tsv'
        val = r'../data/combined34/dev_paired.tsv'
   
    # random

    # back translation
   
    ## llm

    else:
        print(f"train_on:{train_on_}")
        raise ('error : load train fail')

    if using_imdb_fordev is not None and using_imdb_fordev=='imdb1k':
        val = './data/combined34/imdb1k.tsv'

    return train,test,val

def load_test_dataset(test_type):
    if test_type == 'test_rev':
        load_revise_test_data = '../data/revised17/test.tsv'
        return load_revise_test_data
    elif test_type =='test_ori':
        load_ori_test_data = '../data/original17/test.tsv'
        return load_ori_test_data
    elif test_type =='test_comb':
        load_comb_test_data = '../data/combined34/test.tsv'
        return load_comb_test_data


def tokenize_data(tokenizer,data,max_leng=350):
    input_ids = []
    attention_masks = []
    labels = []
    for text, label in data:
        encoded_text = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_leng,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        # print(encoded_text)
        input_ids.append(encoded_text['input_ids'])
        attention_masks.append(encoded_text['attention_mask'])
        labels.append(label)

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    return input_ids, attention_masks, labels

def preprocess_data(file_path,str2id={'Positive':1,"Negative":0},label_text=[0,1],keep_ori=False,printlog=True,batch_num=None):
    # 数据预处理函数
    data = []
    file_end = file_path.split('.')[-1].strip()
    if  file_end=='tsv':
        del_ = '\t'
    elif file_end =='csv':
        del_ = ','
    else:
        print('error')
    if printlog:
        print(f"pre_data del_:{del_}")
    with open(file_path, 'r',newline='', encoding='utf-8') as file:
        tsvr = csv.reader(file,delimiter=del_)
        for row in tsvr:
            if len(row)>=2:
                label,textcf = row[label_text[0]],row[label_text[-1]]
                if label not in str2id.keys():
                    continue
                if keep_ori:
                    data.append((label,row[1],textcf))
                else:
                    data.append((textcf, int(str2id[label])))

        file.close()
    if printlog:
        print(f'len of data: {len(data)}')
    if batch_num is None:
        return data
    else:
        negative_samples = [item for item in data if item[-1] == 0]
        positive_samples = [item for item in data if item[-1] == 1]
        # 随机选择n/2个negative和n/2个positive的元组，  index 一致，考虑到comb时 能取到ori--rev条数的数据
        min_len = min(len(negative_samples), len(positive_samples))
        selected_index  = random.sample(range(min_len), batch_num // 2)
        selected_samples = []
        for i in selected_index:
            selected_samples.append(negative_samples[i])
            selected_samples.append(positive_samples[i])
        return selected_samples

def  appply_test_on_x(model=None, tokenizer=None,max_length=350,device_id=0,batch_size=16,model_path = './sentiment_model_oridb/',test_type='test_rev',str2id={'Negative':0,'Positive':1},printlog=True):
    #### model_path :指定 训练好的模型，sentiment_model_oridb、sentiment_model_revdb
    ### test_type: 指定 测试的数据，ori_test_rev（ori训练的模型测试数据），rev_test_ori（训练的模型训练测试原始数据）
    if tokenizer is None:
        # 指定tokenizer的名称或路径
        tokenizer_path = 'bert-base-uncased'  # 也可以是自定义tokenizer的路径
        # 使用from_pretrained方法加载tokenizer
        tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    # 准备测试数据
    test_data = preprocess_data(load_test_dataset(test_type),str2id,printlog=printlog)
    test_input_ids ,test_attention_masks,test_labels= tokenize_data(tokenizer, test_data,max_leng=max_length)
    test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)


    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 将模型移至GPU（如果可用）
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    if model is None:
        model = BertForSequenceClassification.from_pretrained(model_path, num_labels=2)
    model.to(device)

    model.eval()
    correct = 0
    total =0
    for batch in test_dataloader:
        ba =tuple(t.to(device) for t in batch)
        inputs = {'input_ids': ba[0],
                  'attention_mask':ba[1],
                  'labels':ba[2]}

        with torch.no_grad():
            outputs = model(**inputs,output_hidden_states=True)
        logits = torch.sigmoid(outputs.logits)
        predicted_labes = (torch.argmax(logits,dim=1)>0.5).long()
        correct +=(predicted_labes==ba[2]).sum().item()
        total += len(ba[2])
    accuracy_test = correct/total
    print(f"{test_type}: accuarcy= {accuracy_test}")
    return accuracy_test*100

def load_test_model(test_on_):
    if test_on_=='pre_trained_ori':
        model_path = './sentiment_model_oridb2/'
    elif test_on_ == 'pre_trained_rev':
        model_path ='./sentiment_model_revdb/'
    elif test_on_ == 'pre_trained_comb':
        model_path = './sentiment_model_combdb/'

    elif test_on_ == 'raw_ori':
        model_path = 'sentiment_raw_bert_oridb.pth'
    elif test_on_ == 'raw_rev':
        model_path = 'sentiment_raw_bert_revdb.pth'
    elif test_on_ == 'raw_comb':
        model_path = 'sentiment_raw_bert_combdb.pth'
    elif test_on_ =='raw_distil':
        model_path = 'sentiment_raw_bert_mydistildatadb.pth'
    return  model_path

def load_ood_dataset(ood='yelp'):
    ### return name,text filed name
    if ood=='yelp':
        # https://huggingface.co/datasets/yelp_polarity
        return 'yelp_polarity','text','label'
    elif ood =='amazon':
        # https://huggingface.co/datasets/amazon_polarity
        return 'amazon_polarity','content','label'
    elif ood =='amazon2':
        return "Siki-77/amazon6_5core_polarity",'context','label'
    elif ood =='amazon3':
        return "Siki-77/amazon6_polarity",'context','feeling'
    # elif ood == 'imdb':
    #     return 'imdb','text','label' #不确定为什么报错
    # elif ood == 'imdb':
    #     return 'ajaykarthick/imdb-movie-reviews','review','label' #太小了
    elif ood == 'imdb':
        return "Lucylulu/imdb",'text','label'

    elif ood == 'twitter':
        # 'https://huggingface.co/datasets/carblacac/twitter-sentiment-analysis'
        # return "carblacac/twitter-sentiment-analysis", "text", 'feeling'
        return "Siki-77/twitter2017","text",'feeling'

    elif ood =='hatecheck21':
        return "Siki-77/hatecheck21","test_case","label"
    elif ood =='sst-2':
        #https://huggingface.co/datasets/gpt3mix/sst2/viewer/default/test
        return "Siki-77/sst2","text","label"


def appply_test_on_OOD(model=None,tokenizers=None,max_length=350,device_id=0,test_on_='pre_trained_ori',oodata='amazon',batchsize=16,test_num=None,raw_bert = True):
    #load dataset
    oodata,text,ladbel = load_ood_dataset(oodata)
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    if model is None:
        raw_bert = True
        modelpath = load_test_model(test_on_)

        if test_on_.startswith('raw'):
            model = torch.load(modelpath)
        else:
            raw_bert = False
            model = BertForSequenceClassification.from_pretrained(modelpath, num_labels=2)
    model.to(device)
    model.eval()
    #load dataset
    if test_num is not None:
        test_data = load_dataset(oodata,split=f'test[:{int(test_num)}]')
    else:
        test_data = load_dataset(oodata,split='test')

    # print(f"{test_on_}: {modelpath}")
    print(f"oodata:{oodata} - {f'all:{len(test_data)}' if load_ood_dataset is None else test_num}")
    print(f"test example 1: {test_data[0]}")
    str2id = {'Positive':1,'Negative':0}

    # batchize
    if tokenizers is None:
        tokenizers = AutoTokenizer.from_pretrained('bert-base-uncased')
    def tokenize_batch(batch):
        inputs = tokenizers(batch[text],
                            padding='max_length',
                            max_length=max_length,
                            truncation=True,
                            return_tensors='pt')

        labels = torch.tensor(batch[ladbel])

        # 返回PyTorch张量
        return {
            'input_ids':inputs['input_ids'],
            'attention_mask':inputs['attention_mask'],
            'labels': labels
        }
    encoded_data = test_data.map(tokenize_batch,batched=True)

    encoded_tesor = TensorDataset(torch.tensor(encoded_data['input_ids']),torch.tensor(encoded_data['attention_mask']), torch.tensor(encoded_data['labels']))
    test_dataloader = DataLoader(encoded_tesor,batch_size=batchsize,shuffle=False)

    # test
    total = 0
    correct = 0
    for batch in test_dataloader:
        input_ids ,attention_mask,label = [item.to(device) for item in batch]
        if raw_bert:
            with torch.no_grad():
                outputs = model(input_ids,attention_mask = attention_mask)
            logits = torch.sigmoid(outputs['logits'])
        else:
            inputs = {'input_ids': input_ids,
                      'attention_mask': attention_mask,
                      'labels': label}
            with torch.no_grad():
                outputs = model(**inputs,output_hidden_states=True)
            logits = torch.sigmoid(outputs.logits)
            # if oodata=='imdb':
            #     print(f"{oodata} logit: {logits}")
        pred_labels = (torch.argmax(logits, dim=1) > 0.5).long()
        # if oodata=='imdb':
        #     print(f"{oodata} pred:{pred_labels}\n true:{label}")
        correct +=(pred_labels==label).sum().item()
        total += len(label)
    test_acc = correct/total
    print(f'{oodata}-{test_num} accuarcy = {test_acc*100:.4f}%')
    print('---------------\n')
    return test_acc*100

def run_for_moreoodtest_num(trsmodel=None,model_name='bert-base-uncased',device_id=1,save_best_model = 'best_modelauto.pth',oodama='amazon',test_num_ood=5000):
    # 加载BERT模型和分词器#return_dict=True
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if trsmodel is None:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, return_dict=True)
        # 加载新参数
        model.load_state_dict(torch.load(save_best_model))
    else:
        model = trsmodel
    device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    imdbood,amaood, yelpood,  twiood=0,0,0,0
    model.eval()
    orit = appply_test_on_x(model=model, tokenizer=tokenizer, test_type='test_ori', device_id=device_id)
    revt = appply_test_on_x(model=model, tokenizer=tokenizer, test_type='test_rev', device_id=device_id)
    amaood = appply_test_on_OOD(model=model, tokenizers=tokenizer, oodata=oodama, test_num=test_num_ood, raw_bert=False, device_id=device_id)
    amaood3 = appply_test_on_OOD(model=model, tokenizers=tokenizer, oodata='amazon3', test_num=test_num_ood, raw_bert=False, device_id=device_id)
    yelpood = appply_test_on_OOD(model=model, tokenizers=tokenizer, oodata='yelp', test_num=test_num_ood, raw_bert=False, device_id=device_id)
    imdbood = appply_test_on_OOD(model=model, tokenizers=tokenizer, oodata='imdb', test_num=test_num_ood,  raw_bert=False, device_id=device_id)
    twiood = appply_test_on_OOD(model=model, tokenizers=tokenizer, oodata='twitter', test_num=test_num_ood,
                                raw_bert=False, device_id=device_id)
    sst2ood = appply_test_on_OOD(model=model, tokenizers=tokenizer, oodata='sst-2', test_num=None,
                                         device_id=device_id, raw_bert=False)
    
    return (orit, revt, amaood,amaood3, yelpood,twiood,sst2ood,imdbood)


def load_for_Data_imdb(data_set='imdb',base_model = 'bert-base-uncased',max_length=350,batch_size=32):
    # 下载IMDB数据集
    data_name,text,label = load_ood_dataset(data_set)
    data =load_dataset(data_name)
    # 数据预处理
    tokenizer = AutoTokenizer.from_pretrained(base_model)

    def preprocess_function(examples):
        return tokenizer(examples[text], padding='max_length', truncation=True, max_length=max_length,return_tensors='pt')

    token_data = data.map(preprocess_function, batched=True)
    test_dataset = token_data['test']
    train_dataset = token_data['train']
    # 划分训练集、验证集和测试集
    ds = test_dataset.train_test_split(test_size=0.2,shuffle=True)
    test_dataset, val_dataset = ds['train'],ds['test']
    print(f"train dataset:{train_dataset}")
    print(f"val_dataset:{val_dataset}")
    print(f"test_dataset:{test_dataset}")
    # 创建DataLoader
    def create_dataloader(features):
        dataset = TensorDataset(torch.tensor(features['input_ids']), torch.tensor(features['attention_mask']), torch.tensor(features[label]))
        # features = {k: torch.tensor(v) for k, v in data.items()}
        # dataset = TensorDataset(features['input_ids'], features['attention_mask'], features['label'])
        return DataLoader(dataset, batch_size=batch_size,shuffle=True)

    train_dataloader = create_dataloader(train_dataset)
    val_dataloader = create_dataloader(val_dataset)
    test_dataloader = create_dataloader(test_dataset)
    return train_dataloader,val_dataloader,test_dataloader
