import torch
import torch.nn as nn
import os

from torch.autograd import Function
import numpy as np
from networks.resnet import resnet50
from networks.hkr import *

"""
Reverse Layer component
"""
class ReverseLayer(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

"""
self-attention mechanism: score map * feature
"""
class Conv2d1x1(nn.Module):
    def __init__(self, in_f, hidden_dim, out_f):
        super(Conv2d1x1, self).__init__()
        self.fc1 = nn.Linear(in_f, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_f)
        self.fc3 = nn.Linear(in_f, out_f)
        self.lkrelu = nn.LeakyReLU()

    def forward(self, x):
        
        att = x

        att1 = self.fc1(att)
        att2 = self.lkrelu(att1)
        score_map = self.fc2(att2)
        score_map = F.softmax(score_map, dim = -1)

        out = self.fc3(x)
        attention = torch.mul(score_map, out)

        x = out + attention
        x = self.lkrelu(x)

        return x

class Head(nn.Module):
    def __init__(self, in_f, out_f):
        super(Head, self).__init__()
        #self.do = nn.Dropout(0.2)
        self.mlp = nn.Sequential(nn.Linear(in_f, out_f))

    def forward(self, x):
        #bs = x.size()[0]
        #x_feat = self.pool(x).view(bs, -1)
        x = self.mlp(x)
        #x = self.do(x)
        return x #, x_feat


class ODDN(nn.Module):
    def name(self):
        return 'ODDN'

    def __init__(self, opt):
        super(ODDN, self).__init__()

        self.opt = opt
        self.total_steps = 0
        self.isTrain = opt.isTrain
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.device = torch.device('cuda')
        
        self.HSIC = HKRPair(weights=[0.1, 0.25, 0.5, 0.0, 0.0, 0.0], n_modality=2, sigma=6.0)
        self.gamma = opt.gamma

        self.encoder_feat_dim = 2048
        self.num_classes = 1
        self.loss_fn = torch.nn.BCEWithLogitsLoss()

        self.head_cmp = Head(
            in_f=self.encoder_feat_dim, 
            out_f=self.num_classes
        )
        self.head_tf = Head(
            in_f=self.encoder_feat_dim,
            out_f=self.num_classes
        )

        self.block_cmp = Conv2d1x1(
            in_f = self.encoder_feat_dim,
            hidden_dim=self.encoder_feat_dim // 2, 
            out_f=self.encoder_feat_dim
        )
        self.block_tf = Conv2d1x1(
            in_f=self.encoder_feat_dim, 
            hidden_dim=self.encoder_feat_dim // 2, 
            out_f=self.encoder_feat_dim 
        )

        if self.isTrain and not opt.continue_train:
            self.resnet = resnet50(pretrained=True)
            self.resnet.fc = nn.Linear(2048, 1024)

        if not self.isTrain or opt.continue_train:
            self.resnet = resnet50(num_classes=1024)

        self.resnet = torch.nn.DataParallel(self.resnet)

    def forward(self, input, train = False, label = None, alpha = None):

        ft1, ft2, ft3, ft4, resnet_feat, _ = self.resnet(input)
        tf_feat = self.block_tf(resnet_feat)
        out_tf = self.head_tf(tf_feat)

        tf_loss, cmp_loss, dis_loss = None, None, None
        if train:
            tf_label_np, mask_label_np = label
            tf_label = torch.tensor(tf_label_np).float().cuda()
            tf_loss = self.loss_fn(out_tf.squeeze(-1), tf_label)

            #HSIC
            ft1_cmp, ft2_cmp, ft3_cmp = ft1[mask_label_np], ft2[mask_label_np], ft3[mask_label_np]
            if sum(mask_label_np) != 0:
                dis_loss = self.gamma * self.HSIC([ft1_cmp, ft2_cmp, ft3_cmp])
                
                #reverse branch
                if alpha is not None:
                    reverse_feat = resnet_feat[mask_label_np]
                    reverse_feat = ReverseLayer.apply(reverse_feat, alpha)
                    cmp_feat = self.block_cmp(reverse_feat)
                    out_cmp = self.head_cmp(cmp_feat)
                    
                    cmp_label = np.zeros(sum(mask_label_np)).astype(bool)
                    cmp_label[sum(mask_label_np) // 2:] = True
                    cmp_label = torch.tensor(cmp_label).float().cuda()
                    cmp_loss = self.loss_fn(out_cmp.squeeze(-1), cmp_label)
                else:
                    cmp_loss = None
                
            
            #TF DIS            
            f_cmp, f_no_cmp = tf_feat[tf_label_np&~mask_label_np], tf_feat[tf_label_np&~mask_label_np]
            t_cmp, t_no_cmp =  tf_feat[~tf_label_np&~mask_label_np], tf_feat[~tf_label_np&~mask_label_np]

            # center of no cmp images
            FNCC = f_no_cmp.mean(dim=0, keepdim=True)  
            TNCC = t_no_cmp.mean(dim=0, keepdim=True)
            
            #center of cmp images
            FCC = f_cmp.mean(dim=0, keepdim=True)  
            TCC = t_cmp.mean(dim=0, keepdim=True)

            if torch.isnan(FNCC).sum() != 0 or torch.isnan(TNCC).sum() != 0 or torch.isnan(FCC).sum() != 0 or \
                torch.isnan(TCC).sum() != 0:
                print("DisFeat exists None")
            else:
                dis_nc = 1.0 / (1 + torch.sqrt(torch.pow(FNCC - TNCC, 2).sum()))
                dis_c = 1.0 / (1 + torch.sqrt(torch.pow(FCC - TCC, 2).sum()))
                if dis_loss is None:
                    dis_loss = dis_nc + dis_c
                else:
                    dis_loss = dis_loss + dis_nc + dis_c
            
        return tf_loss, cmp_loss, dis_loss, out_tf


    def save_networks(self, epoch):
        save_filename = 'model_epoch_%s.pth' % epoch
        save_path = os.path.join(self.save_dir, save_filename)

        # serialize model and optimizer to dict
        state_dict = {
            'model': self.state_dict(),
            'total_steps' : self.total_steps,
        }

        torch.save(state_dict, save_path)

    def adjust_learning_rate(self, optimizer ,min_lr=1e-6):
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.8
            if param_group['lr'] < min_lr:
                return False
        self.lr = param_group['lr']
        print('*'*25)
        print(f'Changing lr from {param_group["lr"]/0.8} to {param_group["lr"]}')
        print('*'*25)
        return True
