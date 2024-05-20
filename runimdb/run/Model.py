# 创建一个模型，将BERT模型和分类头部组合在一起
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from torch import nn
from transformers import BertConfig, BertModel, AutoModel, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM


class CustomOutput:
    def __init__(self,full_features=None,logits=None,loss=None,cls_feature=None):
        self.logits = logits
        self.hidden_states = full_features
        self.cls_hidden_states = cls_feature
        self.loss = loss

class SupConModelv3(nn.Module):
    """ref L55：https://github.com/mariomeissner/lightning-hydra-transformers/blob/8e6cae01c1367bb8731167732f134d20554430eb/src/models/hf_model.py#L55"""
    def __init__(self, model_name  = "bert-base-uncased", num_labels = 2,pooling_strategy='cls',head = None, using_head_clsz=True, feat_dim = 128,ft_type='bitgit'):
        '''
        ###BertForSequenceClassification : bert-base-uncased->pool = cls_pooler,head=None,using_head_clsz=True or pooling= cls,head=None,using_head_clsz=False
        ###DistilbertForSequenceClassification : distil-bert-uncased ->pooling = cls,head=linear,using_head_clasz=True
        ###AlbertForSequenceClassification : albert-base-v2 -> pool = cls_pooler,head=None,using_head_clsz=True
        ###RoBertForSequenceClassification : base_model, specific_pooling, specific_head, specific_head_clsz = 'roberta-base', 'cls', None, False
        :param model_name:
        :param num_labels:
        :param head:
        :param feat_dim:
        :param dim_in:
        :param pooling_strategy: c[:,0],pooler,last_avg(all token except cls), first_last_avg to generate feature
        '''
        super(SupConModelv3, self).__init__()

        self.model = AutoModel.from_pretrained(
            model_name, num_labels=num_labels, return_dict=True)
        # print(self.model)
        ## finetune: BitFit
        if ft_type is not None and ft_type== 'bitfit':
            print('** create SupConModelv3 finetuned by bitfit \n')
            for name,param in self.model.named_parameters():
                if "bias" not in name:
                    param.requires_grad = False
        self.dropout_prob = self.model.config.hidden_dropout_prob
        self.dim_in = self.model.config.hidden_size
        print(f"probdrop:{self.dropout_prob},dim in:{self.dim_in}")
        self.cls_dim_in = self.dim_in
        self.pooling = pooling_strategy
        self.using_head_clsz = using_head_clsz
        # print(dim_in)
        if head == "linear":
            self.head = nn.Linear(self.dim_in, feat_dim,bias=True)
            if self.using_head_clsz:self.cls_dim_in = feat_dim
        elif head == "mlp":
            self.head = nn.Sequential(
                    nn.Linear(self.dim_in, self.dim_in),
                    # nn.Tanh(),
                    nn.ReLU(),
                    nn.Linear(self.dim_in, feat_dim,bias=True)
                    )
            if self.using_head_clsz:self.cls_dim_in = feat_dim
        else:
            self.head = lambda x:x

        self.classifier = nn.Linear(self.cls_dim_in, num_labels, bias=True)
        self.model._init_weights(self.classifier) #init
        self.dropout = nn.Dropout(p=self.dropout_prob, inplace=False)

        self.loss = torch.nn.CrossEntropyLoss()
        print(self.classifier.weight.data)
        print(self.classifier.bias.data)


    def forward(self, **kwargs):
        inputs = {'input_ids':kwargs['input_ids'],'attention_mask':kwargs['attention_mask']}
        output = self.model(**inputs,return_dict=True,output_hidden_states=True)

        # embedding  /bert pooling
        hidden_rep = output.last_hidden_state[:,0,:] #(bts,token_num,768)
        if self.pooling == 'last_avg':
            feature = torch.mean(output.last_hidden_state[:,1:,],dim=1)
        elif self.pooling =='cls_pooler':
            feature = output.pooler_output
        elif self.pooling =='first_last_avg':
            feature = (torch.mean(output.hidden_states[1][:,1:,:],dim=1)+torch.mean(output.hidden_states[-1][:,1:,:],dim=1))/2
        elif self.pooling =='dct':
            dct_hidden_states = torch.fft.fftn(output.last_hidden_state,dim=[2])
            feature = torch.mean(dct_hidden_states,dim=1)
        else: #default last_cls
            feature = hidden_rep
        feature = self.head(feature)  # projector head

        # logits
        if self.using_head_clsz:
            drop_pooler = self.dropout(feature) 
        else:
            pooler = output.pooler_output  ## standard  classfication model
            drop_pooler = self.dropout(pooler)
        logits = self.classifier(drop_pooler)

        # cross entroy loss
        if 'labels' in kwargs:
            loss = self.loss(logits, kwargs['labels'])
        else:
            loss = torch.tensor(0.)
        return CustomOutput(full_features=output,cls_feature=feature,logits=logits,loss=loss)

class SupConSBERTv4(nn.Module):
    def __init__(self,sbert_hfpath='sentence-transformers/multi-qa-distilbert-cos-v1',num_labels=2,dim_in=768,ft_type='bitfit'):
        super(SupConSBERTv4, self).__init__()
        self.sbert_model = AutoModel.from_pretrained(sbert_hfpath)
        # bitfit
        if ft_type is  not None and ft_type =='bitfit':
            print('v4model  bit fit')
            for name,param in self.sbert_model.named_parameters():
                if 'bias' not in name:
                    param.requires_grad = False


        self.classifier = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.Tanh(),
            nn.Dropout(p=0.1, inplace=False),
            nn.Linear(dim_in, num_labels, bias=True),
        )
        self.loss = torch.nn.CrossEntropyLoss()

    # Mean Pooling - Take average of all tokens
    def mean_pooling(self,model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self,**kwargs):
        
        encoded_input =  {'input_ids':kwargs['input_ids'],'attention_mask':kwargs['attention_mask']}
        features = self.sbert_model(**encoded_input,return_dict=True)
        # print(features)
        embeddings = self.mean_pooling(model_output=features,attention_mask=kwargs['attention_mask'])
        embeddings = torch.nn.functional.normalize(embeddings,p=2,dim=1)
        # print(embeddings)
        # print(embeddings.shape)
        laebls = kwargs['labels']
        logit = self.classifier(embeddings)
        loss = self.loss(logit,laebls)

        return CustomOutput(cls_feature=embeddings,logits=logit,loss=loss)

class SupConT5v5(nn.Module):
    def __init__(self, model_name, num_labels=2):
        super(SupConT5v5, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)  # 假设是二分类任务
        # print(self.model)

    def forward(self,**kwargs):
        
        outputs = self.model(**kwargs)
        # print(outputs)
        hidden_state = outputs.decoder_hidden_states[-1][:, 0, :]
        # print(f"hidden_state:{hidden_state.shape}")
        logits = outputs.logits
        loss = outputs.loss
        return CustomOutput(cls_feature=hidden_state, logits=logits, loss=loss)

class EarlyStopping:
    def __init__(self, patience=5, verbose=False,delta=0,save_path=''):
        self.patience = patience  
        self.verbose = verbose  # print early stopping information 
        self.counter = 0  
        self.delta = delta # threthold
        self.best_val_loss = float('inf') 
        self.early_stop = False  # control the stopping button
        self.stop_epoch =0
    def __call__(self, val_loss):
        if val_loss >= self.best_val_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print("Early stopping")
        else:
            self.best_val_loss = val_loss
            self.counter = 0

        return self.early_stop
