# 创建一个模型，将BERT模型和分类头部组合在一起
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from torch import nn
from transformers import BertConfig, BertModel, AutoModel, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM


class CustomPCA(nn.Module):
    """ https: // zhuanlan.zhihu.com / p / 373754570"""
    def __init__(self,n_components=128):
        super(CustomPCA, self).__init__()

        self.n_components = n_components

    def forward(self,feature,kernel= None,bias = None):
        """
        :param feature: cuda: tensor bts*768
        :return: PCA 降维到self.n_components的特征： cuda tensor bts*128
        """

        # 计算kernel和bias,最后的变换：y = (x + bias).dot(kernel)
        vecs = feature.cpu().detach().numpy()
        bias = vecs.mean(axis=1, keepdims=True)
        cov = np.cov(vecs.T) # \sum 正定举证
        # 进行svg分解，在cpu上进行
        u, s, vh = np.linalg.svd(cov) #进行svg分解
        W = np.dot(u, np.diag(s ** 0.5))
        W = np.linalg.inv(W.T)
        W = W[:,:self.n_components]

        # GPU 应用变换，然后标准化
        # print(f"feature.device: {feature.device}")
        kernel = torch.tensor(W,dtype=torch.float64).to(feature.device)
        bias = torch.tensor(bias,dtype=torch.float64).to(feature.device)
        if not (kernel is None or bias is None):
            trs_feature = torch.matmul((feature - bias),kernel)
        norm_feature = trs_feature / torch.sqrt((trs_feature**2).sum(dim=1,keepdims=True))
        return norm_feature


class CustomOutput:
    def __init__(self,full_features=None,logits=None,loss=None,cls_feature=None):
        self.logits = logits
        self.hidden_states = full_features
        self.cls_hidden_states = cls_feature
        self.loss = loss
class SupConModelv1(nn.Module):
    """made by yongjie"""
    def __init__(self, model_name  = "distilbert-base-uncased", num_labels = 2, head = "mlp", feat_dim = 128, dim_in = 768):

        super(SupConModelv1, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, num_labels = num_labels)

        dim_in = self.model.config.hidden_size
        # print(dim_in)
        if head == "linear":
            self.head = nn.Linear(dim_in, feat_dim,bias=False)
        elif head == "mlp":
            self.head = nn.Sequential(
                    nn.Linear(dim_in, dim_in),
                    nn.Tanh(),
                    nn.Linear(dim_in, feat_dim,bias=False)
                    )
        else:
            raise NotImplementedError(
                    "head not supported: {}".format(head))
        # self.classifier = nn.Sequential(
        #         nn.Linear(dim_in, dim_in),
        #         nn.ReLU(),
        #         nn.Dropout(p=0.2),
        #         nn.Linear(dim_in, num_labels),
        #         )
        # v1 classifier :bert for sequence model
        self.classifier = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.Tanh(),
            nn.Dropout(p=0.1,inplace=False),
            nn.Linear(dim_in, num_labels,bias=True),
        )
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, **kwargs):

        output = self.model(**kwargs)
        hidden_rep = output.hidden_states[-1][:, 0, :]
        feature = self.head(hidden_rep)
        logits = self.classifier(hidden_rep)
        # logits = self.classifier(feature)
        loss = self.loss(logits,kwargs['labels'])
        return CustomOutput(cls_feature=feature,logits=logits,loss=loss)

class SupConModelv2(nn.Module):
    """参考L55：https://github.com/mariomeissner/lightning-hydra-transformers/blob/8e6cae01c1367bb8731167732f134d20554430eb/src/models/hf_model.py#L55"""
    def __init__(self, model_name  = "bert-base-uncased", num_labels = 2,head = None,using_head_clsz=False, feat_dim = 128, dim_in = 768):
        '''
        :param model_name:
        :param num_labels:
        :param head:
        :param feat_dim:
        :param dim_in:
        :param pooling_strategy: c[:,0],pooler,last_avg(all token except cls), first_last_avg to generate feature
        '''

        super(SupConModelv2, self).__init__()
        self.model = AutoModel.from_pretrained(
            model_name, num_labels=num_labels, return_dict=True)

        self.dim_in = self.model.config.hidden_size
        self.using_head_clsz = using_head_clsz
        # print(dim_in)
        if head == "linear":
            self.head = nn.Linear(dim_in, feat_dim,bias=False)
            if self.using_head_clsz:self.cls_dim_in = feat_dim
        elif head == "mlp":
            self.head = nn.Sequential(
                    nn.Linear(dim_in, dim_in),
                    # nn.Tanh(),
                    nn.ReLU(),
                    nn.Linear(dim_in, feat_dim,bias=False)
                    )
            if self.using_head_clsz:self.cls_dim_in = feat_dim
        else:
            self.head = lambda x:x
            self.cls_dim_in = self.dim_in

        # self.classifier = nn.Linear(self.cls_dim_in, num_labels, bias=True)
        # self.model._init_weights(self.classifier) #init
        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.cls_dim_in, out_features=feat_dim),
            nn.ReLU(),
            nn.Linear(in_features=feat_dim, out_features=num_labels)
        )
        self.model._init_weights(self.classifier1)
        self.dropout = nn.Dropout(p=0.1, inplace=False)
        self.loss = torch.nn.CrossEntropyLoss()


    def forward(self, **kwargs):
        inputs = {'input_ids':kwargs['input_ids'],'attention_mask':kwargs['attention_mask']}
        output = self.model(**inputs,return_dict=True,output_hidden_states=True)
        # embedding
        hidden_rep = output.last_hidden_state[:,0,:] #(bts,token_num,768)
        feature = self.head(hidden_rep)  # projector head

        # logits
        pooler = output.pooler_output  ## bertcls 是直接拿pooler output最下游分类的
        pooler = self.dropout(pooler)
        logits = self.classifier(pooler)

        # cross entroy loss
        loss = self.loss(logits, kwargs['labels'])
        return CustomOutput(cls_feature=feature,logits=logits,loss=loss)
class SupConModelv3(nn.Module):
    """参考L55：https://github.com/mariomeissner/lightning-hydra-transformers/blob/8e6cae01c1367bb8731167732f134d20554430eb/src/models/hf_model.py#L55"""
    def __init__(self, model_name  = "bert-base-uncased", num_labels = 2,pooling_strategy='cls',head = None, using_head_clsz=True, feat_dim = 128,ft_type='bitgit',linearlayer=1):
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
        ## 冻结 .. BitFit
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
        elif head == 'pca':
            self.head= CustomPCA(n_components=feat_dim)
            if self.using_head_clsz: self.cls_dim_in = feat_dim
        else:
            self.head = lambda x:x

        if linearlayer<2:
            self.classifier = nn.Linear(self.cls_dim_in, num_labels, bias=True)
            self.model._init_weights(self.classifier) #init
        else:
            self.classifier = nn.Sequential(
                nn.Linear(in_features=self.cls_dim_in, out_features=feat_dim,bias=True),
                nn.Tanh(),
                nn.Linear(in_features=feat_dim, out_features=num_labels,bias=True)
            )
            self.init_and_print_weights()
       
        self.dropout = nn.Dropout(p=self.dropout_prob, inplace=False)

        self.loss = torch.nn.CrossEntropyLoss()
        # print(self.classifier.weight.data)
        # print(self.classifier.bias.data)
       
    
    def init_and_print_weights(self):
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
                # print(layer.weight.data)
                print(layer.bias.data)

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
            drop_pooler = self.dropout(feature) ##尝试用变换后feature去分类，不是很好
        else:
            pooler = output.pooler_output  ## bertcls 是直接拿pooler output最下游分类的
            drop_pooler = self.dropout(pooler)
        logits = self.classifier(drop_pooler)

        # cross entroy loss
        loss = self.loss(logits, kwargs['labels'])
        return CustomOutput(cls_feature=feature,logits=logits,loss=loss)

class SupConSBERTv4(nn.Module):
    def __init__(self,sbert_hfpath='sentence-transformers/multi-qa-distilbert-cos-v1',num_labels=2,dim_in=768,feat_dim=128,ft_type='bitfit',linearlayer=1):
        super(SupConSBERTv4, self).__init__()
        self.sbert_model = AutoModel.from_pretrained(sbert_hfpath)
        # bitfit
        if ft_type is  not None and ft_type =='bitfit':
            print('v4model  bit fit')
            for name,param in self.sbert_model.named_parameters():
                if 'bias' not in name:
                    param.requires_grad = False

        if linearlayer >1:
             self.classifier = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.Tanh(),
                nn.Dropout(p=0.1, inplace=False),
                nn.Linear(dim_in, feat_dim, bias=True),
                nn.Tanh(),
                nn.Linear(feat_dim, num_labels, bias=True),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.Tanh(),
                nn.Dropout(p=0.1, inplace=False),
                nn.Linear(dim_in, num_labels, bias=True),
            )
        self.init_and_print_weights()
        self.loss = torch.nn.CrossEntropyLoss()
    
    def init_and_print_weights(self):
        cur_layer = 0
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias, 0)
                # print(layer.weight.data)
                cur_layer +=1
                if len(layer.bias.data)<10:
                    print(layer.bias.data)
        print(f'cls layer :{cur_layer-1}')
    # Mean Pooling - Take average of all tokens
    def mean_pooling(self,model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self,**kwargs):
        #直接放原始句子就可以
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
        # self.t5_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        # self.classifier = nn.Linear(self.t5_model.config.hidden_size, num_labels)
        # self.loss = torch.nn.CrossEntropyLoss()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)  # 假设是二分类任务
        # print(self.model)
    def forward(self,**kwargs):
        # encoded_input = {'input_ids': kwargs['input_ids'], 'attention_mask': kwargs['attention_mask'],'labels':kwargs['labels']}
        # print(kwargs['input_ids'].size())
        # outputs = self.t5_model(**encoded_input)
        # hidden_state = outputs.last_hidden_state[:, 0, :]  # Assuming pooler_output not available
        # logits = self.classifier(hidden_state)
        # lables = kwargs['labels']
        # loss = self.loss(logits, lables)
        outputs = self.model(**kwargs)
        # print(outputs)
        hidden_state = outputs.decoder_hidden_states[-1][:, 0, :]
        logits = outputs.logits
        loss = outputs.loss
        return CustomOutput(cls_feature=hidden_state, logits=logits, loss=loss)

class SupConGSv6(nn.Module):
    def __init__(self, model_name,dim_in=768, num_labels=2):
        super(SupConGSv6, self).__init__()
        self.model_base = model_name
        if model_name.startswith('sentence'):
            self.model = AutoModel.from_pretrained(model_name)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)  # 假设是二分类任务
        
        self.classifier = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.Tanh(),
            nn.Dropout(p=0.1, inplace=False),
            nn.Linear(dim_in, num_labels, bias=True),
        )
        self.loss = torch.nn.CrossEntropyLoss()
        # print(self.model)
        # print(self.model)
    
    def forward(self,**kwargs):
     
        hidden_state=None
       
        if self.model_base.startswith('t5'):
            outputs = self.model(**kwargs)
            hidden_state = outputs.decoder_hidden_states[-1][:, 0, :]
       
        else:
            outputs = self.model(**kwargs)
            hidden_state = outputs.hidden_states[-1][:,0,:]
        laebls = kwargs['labels']
        # print(f"hidden_state:{hidden_state.shape}")
        logits  = self.classifier(hidden_state)
        loss = self.loss(logits,laebls)
        # logits = outputs.logits
        # loss = outputs.loss
        return CustomOutput(cls_feature=hidden_state, logits=logits, loss=loss)

class EarlyStopping:
    def __init__(self, patience=5, verbose=False,delta=0,save_path=''):
        self.patience = patience  # 当验证集损失连续 patience 次没有改善时停止训练
        self.verbose = verbose  # 是否打印早停信息
        self.counter = 0  # 连续没有改善的计数器
        self.delta = delta #损失改善阈值
        self.best_val_loss = float('inf')  # 当前最佳的验证集损失值
        self.early_stop = False  # 是否触发早停
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
