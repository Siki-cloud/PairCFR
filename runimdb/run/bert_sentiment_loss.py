import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

  
class SupConLossv2(nn.Module):

    def __init__(self, num_classes=2, lambda_=0.8, beta=1.,distance="cosine", device=torch.device("cuda"), temperature=0.3,
                 contrast_mode='all', except_netural = True,
                 ):
        super(SupConLossv2, self).__init__()
        self.num_classes = num_classes
        self.lambda_ = lambda_
        self.beeta = beta
        self.distance = distance
        self.device = device
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.except_1 = except_netural

    def forward(self, output, labels, mask=None,no_grad=True):
        if hasattr(output,'cls_hidden_states'):
            features = output.cls_hidden_states
        else:
            features= output.hidden_states[-1][:,0,:]
        # features = output.logits

        """
        Compute loss for model. 
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.

        """

        if len(features.shape) < 2:
            raise ValueError("feature needs to be [batch_size, feat_len]")

        if len(labels.shape) < 2:
            labels = labels.view(labels.shape[0], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        elif labels is not None:
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(self.device)
        else:
            mask = mask.float().to(self.device)

        # compute the loss over contrastive loss
        if self.distance == "cosine":
            norm_features = F.normalize(features,p=2,dim=1)
            distance = torch.matmul(norm_features,norm_features.T)
        elif self.distance == "eclidean":
            distance = torch.matmul(features, features.T)
        else:
            raise ValueError("No implementation on specified metrics")

        anchor_dot_contrast = torch.div(
            distance,
            self.temperature)
        # for numerical stability
        contrast_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        anchor_dot_contrast = anchor_dot_contrast - contrast_max.detach()
        contrast_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(self.device),
            0
        )
        pos_mask = mask * contrast_mask
        neg_mask = (~(mask==1)).float() * contrast_mask
        
        # 3-classification for NLI # 0 1 2
        if (labels == 2).any().item() and self.except_1:   
            mask_neutural_index =  (labels == 1).nonzero()  
            mask_neutural_index = mask_neutural_index[:,0]
            mask_neutural_False =  torch.ones((labels.size(0),), dtype=torch.bool)
            mask_neutural_False[mask_neutural_index] = False
            mask_neutural =  torch.unsqueeze(mask_neutural_False, 0).repeat(labels.size(0), 1)
         
            mask_neutural = mask_neutural.float().to(self.device)
            neg_mask = neg_mask * mask_neutural #  filter example with neutural label
            neg_mask[mask_neutural_index ,:]= 0.

        exp_contrast = torch.exp(anchor_dot_contrast)* neg_mask # only negative pair 
        log_prob = anchor_dot_contrast - torch.log(exp_contrast.sum(1, keepdim=True)+torch.exp(anchor_dot_contrast))


        # compute mean of log-likelihood over positive
        mean_log_prob_pos = -(pos_mask * log_prob).sum(1) / (pos_mask.sum(1)+1e-8)

        # loss
        contrastive_loss = mean_log_prob_pos.mean()

        # Calculate the cross entropy loss
        ce_loss = output.loss
        total_loss = self.lambda_ * contrastive_loss + self.beeta * ce_loss
        if torch.isnan(total_loss):
            print(f"total loss: nan\n"
                  f"true label :{labels}\n"
                  f"celoss:{ce_loss}\n"
                  f"conloss:{contrastive_loss}\n"
                  f"mean_log_prob_pos:{mean_log_prob_pos}\n"
                  f"log_prob:{log_prob}\n"
                  f"exp_contrast:{exp_contrast}\n"
                  f"neg_mask:{neg_mask}\n"
                  f"pos_maskL{pos_mask}\n"
                  f"anchor_dot_contrast:{anchor_dot_contrast}")
        return total_loss,ce_loss,contrastive_loss
