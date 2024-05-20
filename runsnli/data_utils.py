import traceback
import pandas as pd
import torch
from torch.utils.data import DataLoader, Sampler, TensorDataset
from torch.utils.data import Dataset, DataLoader
import random
from datasets import load_dataset


def load_train_data_file_name(train_type:str):
    if train_type.startswith('original'):
        train = r"./data/ori16/train.tsv"
        test = r"./data/ori16/test.tsv"
        dev = r"./data/ori16/dev.tsv"
    
    elif train_type =='ori_rh_rp83' or train_type.startswith('synori_rh_rp'):
        train = r"./data/ori_rh_rp83/train.tsv"
        test = r"./data/ori_rh_rp83/test.tsv"
        dev = r"./data/ori_rh_rp83/dev.tsv"
    
    elif train_type =='snli':
        train = r"../data/snli/train.tsv"
        test = r"../data/snli/test.tsv"
        dev = r"../data/snli/dev.tsv"
    return train, dev, test

# Data preprocessing function
def preprocess_function(data,tokenizer,max_length,text1='sentence1',text2='sentence2',glod_label='gold_label'):
        try:
            tokenized = tokenizer(data[text1].tolist(), data[text2].tolist(), padding='max_length', truncation=True,
                                  max_length=max_length,
                                  return_tensors='pt')
            input_ids = tokenized['input_ids']
            attention_mask = tokenized['attention_mask']
            label_map = {'entailment': 0, 'neutral': 1, 'contradiction': 2, '1': 1, '0': 0, '2': 2, 1: 1, 0: 0, 2: 2}
            label_ = data[glod_label].map(label_map).tolist()
            # print(label_)
            labels = torch.tensor(label_)
            _dataset = TensorDataset(input_ids, attention_mask, labels)
            return _dataset
        except Exception as e:
            for i,j,k in zip( data[text1].tolist(), data[text2].tolist(),data[glod_label]):
                if not isinstance(i,str) or not isinstance(j,str):
                    print(f"s1:{i},s2:{j}, k:{k}")
            traceback.print_exc(e)


def snli_add_data(data_source,add_n = 1000,type_='train'):
    # loading SNLI dataset
    snli_dataset = load_dataset("Siki-77/snli_filter-1")
    num_samples = len(snli_dataset[type_])
    # random sample n instances
    indices = random.sample(range(num_samples), int(add_n))
    selected_snli_data = snli_dataset[type_].select(indices)
    not_1_selected = []
    for datap in selected_snli_data:
        if datap['label']=='-1' or datap['label']== -1:
            print(f"label_minus_1:{datap}")
        elif  isinstance(datap['premise'],str) and isinstance(datap['hypothesis'],str):
            not_1_selected.append(datap)

    # transfer to Pandas DataFrame
    selected_snli_df = pd.DataFrame(not_1_selected)
    # rename ("premise","hypothesis"),"label"
    selected_snli_df = selected_snli_df.rename(columns={'premise': 'sentence1','hypothesis':'sentence2','label':'gold_label'})  # 根据需要修改列名
    # concat the train_data
    data_source = pd.concat([data_source, selected_snli_df], ignore_index=True)

    print(f"**after adding for {type_}:{len(data_source)}\n")

    return data_source

def load_data_for_train_full(train_type,tokenizer,max_length=64,bts=16):
    # train_type : snli
    data_pth, sentence_pair,label_col, _ = load_oodata_colname(train_type)
    dataset = load_dataset(data_pth)

    # construct train,val,test set. 
    train_data = dataset["train"]
    val_data = dataset["validation"]
    test_data = dataset["test"]

    # tranfer the data to DataFrame, delete the empty rows
    train_df = pd.DataFrame(train_data).dropna()
    val_df = pd.DataFrame(val_data).dropna()
    test_df = pd.DataFrame(test_data).dropna()
    print(f"train len:{train_df.shape[0]}")
    print(f"dev len:{val_df.shape[0]}")
    print(f"test len:{test_df.shape[0]}")
    # preprocess
    train_dataset = preprocess_function(train_df,tokenizer=tokenizer,max_length=max_length,text1=sentence_pair[0],text2=sentence_pair[1],glod_label=label_col)
    dev_dataset = preprocess_function(val_df,tokenizer=tokenizer,max_length=max_length,text1=sentence_pair[0],text2=sentence_pair[1],glod_label=label_col)
    test_dataset = preprocess_function(test_df,tokenizer=tokenizer,max_length=max_length,text1=sentence_pair[0],text2=sentence_pair[1],glod_label=label_col)
    # transfer to PyTorch DataLoader
    train_loader = DataLoader(train_dataset, batch_size=bts, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=bts, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=bts, shuffle=False)

    return train_loader, dev_loader, test_loader

def load_data_for_trainsingle(train_type,tokenizer,max_length=50,bts=16,add_data_n=1000,Tr_Te_Dev=[8,2,1],stop_batch_num=None):
    train_file, dev_file, test_file = load_train_data_file_name(train_type=train_type)
    # loading raw data
    train_data = pd.read_csv(train_file, sep='\t', header=0)  # tsv file, splited with '\t'
    dev_data = pd.read_csv(dev_file, sep='\t', header=0)
    test_data = pd.read_csv(test_file, sep='\t', header=0)
    if add_data_n is not None and add_data_n>=1:
        print(f"**ori1.6 loading addintional {add_data_n}")
        train_add_n = int(add_data_n)
        dev_data_n = train_add_n // (Tr_Te_Dev[0]/Tr_Te_Dev[2])
        test_data_n = train_add_n // (Tr_Te_Dev[0]/Tr_Te_Dev[1])
        train_data = snli_add_data(train_data,add_n=train_add_n,type_='train')
        dev_data = snli_add_data(dev_data, add_n=dev_data_n, type_='validation')
        test_data = snli_add_data(test_data, add_n=test_data_n, type_='test')
    
    # Extract specified number of entries, controlling simultaneous presence/removal of every 5 consecutive lines
    if stop_batch_num is not None and stop_batch_num > 0 :
            indexes = list(range(len(train_data)))
            rm_num = len(train_data) - stop_batch_num*bts
            drop_group_index = random.sample(indexes,rm_num)
            newdata = train_data.drop(index=drop_group_index).reset_index(drop=True)        
            train_data =  newdata.copy()
        
    print(f"train len:{len(train_data)}")
    print(f"dev len:{len(dev_data)}")
    print(f"test len:{len(test_data)}")
    # Preprocess the data
    train_dataset = preprocess_function(train_data,tokenizer=tokenizer,max_length=max_length)
    dev_dataset = preprocess_function(dev_data,tokenizer=tokenizer,max_length=max_length)
    test_dataset = preprocess_function(test_data,tokenizer=tokenizer,max_length=max_length)

    # Transfer to PyTorch DataLoader
    train_loader = DataLoader(train_dataset, batch_size=bts, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=bts, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=bts, shuffle=False)

    return train_loader, dev_loader, test_loader

def load_data_for_train_comb(train_type,tokenizer,max_length=50,bts_size=16,bts_n=5,rnd_gc=None,stop_batch_num=None):
    train_file, dev_file, test_file = load_train_data_file_name(train_type=train_type)
    n = bts_n  # Each batch contains consecutive 5 data entries.
    batch_size = bts_size  # batch size
    train_data_size = None
    if stop_batch_num is not None and stop_batch_num > 0:
        train_data_size = batch_size * stop_batch_num
   
    # customized Sampler
    train_dataset = SNLIDataset(train_file, tokenizer, max_length, n=bts_n,rnd_gc=rnd_gc,train_data_size=train_data_size)
    train_custom_sampler = CustomSampler(train_dataset, batch_size, n,rnd_gc=rnd_gc)

    dev_dataset = SNLIDataset(dev_file, tokenizer, max_length, n=bts_n,rnd_gc=rnd_gc)
    dev_custom_sampler =CustomSampler(dev_dataset,bts_size,n,rnd_gc=rnd_gc)

    test_dataset = SNLIDataset(test_file, tokenizer, max_length, n=bts_n)
    test_custom_sampler =CustomSampler(test_dataset,bts_size,n)
    print(f"\nload dataset:\ntrain len:{len(train_dataset)}, dev len:{len(dev_dataset)}, test len:{len(test_dataset)}")

    # create DataLoader，utilze  customized Sampler
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_custom_sampler)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, sampler=dev_custom_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=test_custom_sampler)

    return train_loader, dev_loader, test_loader

class CustomSampler(Sampler):
    def __init__(self, data_source, batch_size, n,rnd_gc=None):
        self.data_source = data_source
        self.batch_size = batch_size
        self.n = n
        if rnd_gc is not None and rnd_gc>0:
            rnd_gc = min(rnd_gc,4)
            self.n = rnd_gc + 1
        self.indices = list(range(len(data_source)))
        self.num_batches = len(self.indices) // self.batch_size

    def __iter__(self):
        # Group the data into sets of n consecutive entries, then shuffle each group individually.
        indices = [self.indices[i:i + self.n] for i in range(0, len(self.indices), self.n)]
        # random.shuffle(indices) # If intra-group shuffling is needed, please uncomment the corresponding line of code.
        indices = [item for sublist in indices for item in sublist]
        # Divide into larger groups, each containing bts elements, and shuffle the groups.
        batches = [indices[i:i + self.batch_size] for i in range(0, len(indices), self.batch_size)]
        random.shuffle(batches)
        indices = [item for sublist in batches for item in sublist]

        return iter(indices[:self.num_batches * self.batch_size])

    def __len__(self):
        return self.num_batches * self.batch_size

class SNLIDataset(Dataset):

    def __init__(self, data_path, tokenizer, max_length,n=5,rnd_gc = None,train_data_size=None):
        self.data = pd.read_csv(data_path, sep='\t', header=0)# skip the headline，if the 1st row is headline
        self.n = n
        # Extract a specified number of data entries while controlling the simultaneous presence or removal of every 5 consecutive lines
        if train_data_size is not None and train_data_size > 0 :
            exist_groups_num = train_data_size // n
            indexes = list(range(len(self.data)))
            indexe_groups =  [indexes[i:i + self.n] for i in range(0, len(self.data), self.n)]
            rm_indexes =  []
            groups_indexes = list(range(len(indexe_groups)))
            drop_down_groups_num = len(indexe_groups) - exist_groups_num
            drop_group_index = random.sample(groups_indexes,drop_down_groups_num)
            drop_indexes_row = []
            for i in drop_group_index:
                sub_groups = indexe_groups[i]
                drop_indexes_row.extend(sub_groups)
            newdata = self.data.drop(index=drop_indexes_row).reset_index(drop=True)        
            self.data =  newdata.copy()
            # print(f'new data:{newdata}')
            print(f'new data {len(newdata)}')
        
        if rnd_gc is not None and rnd_gc > 0:
            rnd_gc = min(rnd_gc,4)
            print(f"select {rnd_gc} cfs from 4 cfs")
            indexes = list(range(len(self.data)))
            indexe_groups =  [indexes[i:i + self.n] for i in range(0, len(self.data), self.n)]
            rm_indexes =  []
            drop_down_row = max(self.n-rnd_gc-1,0)
            for sub_list in indexe_groups:
                sub_ = []
                rm_ = random.sample(sub_list[1:],drop_down_row)
                sub_.extend(rm_)
                rm_indexes.extend(sub_)
                # print(sub_list)
                # print(rm_)
                # print(sub_)
                # raise('eeor')
            newdata = self.data.drop(index=rm_indexes).reset_index(drop=True)        
            self.data =  newdata.copy()
            self.n = rnd_gc+1
        print(f'SNLIDataset:{len(self.data)}')
        
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # implementation here to get item based on index idx
       
        # idx  is the index of a single sample
        data_batch = self.data.iloc[idx]
     
        inputs = self.tokenizer(
            data_batch['sentence1'],
            data_batch['sentence2'],
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        label_map = {'entailment': 0, 'neutral': 1, 'contradiction': 2, '1': 1, '0': 0, '2': 2, 1: 1, 0: 0, 2: 2}
        label_ = label_map[data_batch['gold_label']]
        labels = torch.tensor(label_) 
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



def load_ind_file_name(test_type):
    if  test_type  =='ori':
        file = r'./data/ori16/test.tsv'
    elif test_type == 'rh':
        file = r'./data/ori_rh33/test.tsv'
    elif test_type == 'rp':
        file = r'./data/ori_rp33/test.tsv'
    # RH
    # RP & RH
    return file

def apply_snli_indomain(test_type,model,tokenizer, max_length=50,batch_size=20,device_id=None):
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    test_file = load_ind_file_name(test_type)
    # loading  test dataset 
    test_data = pd.read_csv(test_file, sep='\t', header=0)  


    # preprocess the test dataset
    test_dataset = preprocess_function(test_data,tokenizer=tokenizer,max_length=max_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    total = 0
    correct = 0
    for batch in test_loader:
        batch = tuple(t.to(device) for t in batch)
        input_id, atten_mask, labels = batch[0], batch[1], batch[2]
        inputs = {'input_ids': input_id, 'attention_mask': atten_mask, 'labels': labels}
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        pred_labels = torch.argmax(outputs.logits, dim=1)
        correct += (pred_labels == labels).sum().item()
        total += len(labels)
    test_acc = correct / total
    print(f'{test_type}-{total} accuarcy = {test_acc * 100:.5f}%')
    print('---------------\n')
    return test_acc * 100

def load_oodata_colname(oodname):
    if oodname =='mnli_mm':
        # return ("glue", "mnli"),  ("premise","hypothesis"),"label",('train','validation_mismatched')
        return "SetFit/mnli_mm",('text1','text2'),'label',('train','validation')
    elif oodname =='mnli_m':
        # return ("glue", "mnli"), ("premise","hypothesis"),"label",('train','validation_matched')
        return "SetFit/mnli",('text1','text2'),'label',('train','validation')
    elif oodname == "snli" or oodname.endswith('snli'):
        return "Siki-77/snli_filter-1",("premise","hypothesis"),"label",('train','validation')
    
    elif oodname =='negation':
        return "pietrolesci/stress_tests_nli",("sentence1","sentence2"),"label",('negation','negation')
    elif oodname =='spelling_error':
        return "pietrolesci/stress_tests_nli",("sentence1","sentence2"),"label",('spelling_error','spelling_error')
    elif oodname =='word_overlap':
        return "pietrolesci/stress_tests_nli",("sentence1","sentence2"),"label",('word_overlap','word_overlap')


def apply_snliood_test(ood_name, model=None, tokenizer=None,max_length=50,batch_size=16,test_num=None,device_id=0):
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    # load dataset
    data_pth, sentence_pair,label_col, train_test= load_oodata_colname(oodname=ood_name)
    ## set lenght of ood
    split_test = train_test[-1]
    if test_num is not None:
        split_test = f'{train_test[-1]}[:{int(test_num)}]'
    print(f"**cur:{ood_name},split= {split_test}*")
    ## start load
    if isinstance(data_pth,tuple):
        print(data_pth)
        data = load_dataset(data_pth[0],data_pth[-1],split=split_test)
    if isinstance(data_pth,str):
        print('str')
        data = load_dataset(data_pth,split=split_test)
    else:
        raise ('please specify the ood dataset name')
    ## print log
    if test_num is None:
        test_num = len(data)
    print(f"** TEST ON {ood_name} - load {data_pth} - test_num: {test_num}")

    ### tokenize
    def tokenize_batch(batch):
        inputs = tokenizer(batch[sentence_pair[0]],batch[sentence_pair[-1]],
                            padding='max_length',
                            max_length=max_length,
                            truncation=True,
                            return_tensors='pt')
        labels = torch.tensor(batch[label_col])
        # return PyTorch tensors
        return {'input_ids':inputs['input_ids'],
            'attention_mask':inputs['attention_mask'],
            'labels': labels}
    encoded_data = data.map(tokenize_batch,batched=True)

    encoded_tesor = TensorDataset(torch.tensor(encoded_data['input_ids']),torch.tensor(encoded_data['attention_mask']), torch.tensor(encoded_data['labels']))
    test_dataloader = DataLoader(encoded_tesor,batch_size=batch_size,shuffle=False)
    total = 0
    correct = 0
    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)
        input_id, atten_mask, labels = batch[0], batch[1], batch[2]
        inputs = {'input_ids': input_id, 'attention_mask': atten_mask, 'labels': labels}
        # print(f"{total}:  {labels}")
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        pred_labels =  torch.argmax(torch.softmax(outputs.logits, dim=1),dim=1)
       
        correct += (pred_labels == labels).sum().item()
        total += len(labels)
    test_acc = correct / total
    print(f'{ood_name}-{test_num} accuarcy = {test_acc * 100:.5f}%')
    print('---------------\n')
    return test_acc * 100

# if __name__ == '__main__':
#     # apply_ood_test(ood_name='mnli-mm')