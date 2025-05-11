import os
import torch
import torch.utils.data
import torch.distributed
import torch.backends.cudnn


class prototype_dist_estimator():
    def __init__(self, feature_num ):
        super(prototype_dist_estimator, self).__init__()

        #self.cfg = cfg
        self.class_num = 6 # 根据数据集修改
        #_, backbone_name = cfg.MODEL.NAME.split('_')
        self.feature_num = feature_num
        # momentum 
        self.use_momentum = False   #  根据cfg修改
        self.momentum = 0.9         #  根据cfg修改

        # init prototype   初始原型
        self.init(feature_num=feature_num, resume=None)   #  根据cfg修改
        self.IGNORE_LABEL = 255
        self.OUTPUT_DIR = 'proto'
        

    def init(self, feature_num, resume=""):
        if resume:
            if feature_num == self.cfg.MODEL.NUM_CLASSES:
                resume = os.path.join(resume, 'prototype_out_dist.pth')
            elif feature_num == self.feature_num:
                resume = os.path.join(resume, 'prototype_feat_dist.pth')
            else:
                raise RuntimeError("Feature_num not available: {}".format(feature_num))
            print("Loading checkpoint from {}".format(resume))
            checkpoint = torch.load(resume, map_location=torch.device('cpu'))
            self.Proto = checkpoint['Proto'].cuda(non_blocking=True)
            self.Amount = checkpoint['Amount'].cuda(non_blocking=True)
        else:
            self.Proto = torch.zeros(self.class_num, feature_num).cuda(non_blocking=True) # Proto=7*1024
            self.Amount = torch.zeros(self.class_num).cuda(non_blocking=True)      # Amount = 7*1

    def update(self, features, labels):
        mask = (labels != self.IGNORE_LABEL)
        # remove IGNORE_LABEL pixels
        labels = labels[mask]
        features = features[mask]
        if not self.use_momentum:  # self.use_momentum=false 
            N, A = features.size()
            C = self.class_num
            # refer to SDCA for fast implementation  参考SDCA来快速实现
            features = features.view(N, 1, A).expand(N, C, A)
            onehot = torch.zeros(N, C).cuda()
            onehot.scatter_(1, labels.view(-1, 1), 1)
            NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)
            features_by_sort = features.mul(NxCxA_onehot)
            Amount_CXA = NxCxA_onehot.sum(0)
            Amount_CXA[Amount_CXA == 0] = 1
            mean = features_by_sort.sum(0) / Amount_CXA
            sum_weight = onehot.sum(0).view(C, 1).expand(C, A)
            weight = sum_weight.div(
                sum_weight + self.Amount.view(C, 1).expand(C, A)
            )
            weight[sum_weight == 0] = 0
            self.Proto = (self.Proto.mul(1 - weight) + mean.mul(weight)).detach()
            self.Amount = self.Amount + onehot.sum(0)
        else:
            # momentum implementation
            ids_unique = labels.unique()
            for i in ids_unique:
                i = i.item()
                mask_i = (labels == i)
                feature = features[mask_i]
                feature = torch.mean(feature, dim=0)
                self.Amount[i] += len(mask_i)
                self.Proto[i, :] = self.momentum * feature + self.Proto[i, :] * (1 - self.momentum)
        
    def save(self, name):
        torch.save({'Proto': self.Proto.cpu(),
                    'Amount': self.Amount.cpu()
                    },
                   os.path.join(self.OUTPUT_DIR, name))
