import torch
import torch.nn as nn
import torch.nn.functional as F

# 更改特征-整体原型损失 为 单图原型-整体原型损失

class PrototypeContrastiveLoss(nn.Module):  
    def __init__(self):
        super(PrototypeContrastiveLoss, self).__init__()
        #self.cfg = cfg
        self.IGNORE_LABEL = 255
        self.CONTRAST_TAU = 50.0

    def forward(self, Proto, feat, labels):
        """
        Args:
            C: NUM_CLASSES A: feat_dim B: batch_size H: feat_high W: feat_width N: number of pixels except IGNORE_LABEL
            Proto: shape: (C, A) the mean representation of each class 
            feat: shape (BHW, A) -> (N, A)
            labels: shape (BHW, ) -> (N, )

        Returns:

        """
        assert not Proto.requires_grad
        assert not labels.requires_grad
        assert feat.requires_grad
        assert feat.dim() == 2
        assert labels.dim() == 1
        '''
        # remove IGNORE_LABEL pixels
        mask = (labels != self.IGNORE_LABEL)
        labels = labels[mask]
        feat = feat[mask]

        feat = F.normalize(feat, p=2, dim=1)
        Proto = F.normalize(Proto, p=2, dim=1)

        logits = feat.mm(Proto.permute(1, 0).contiguous())
        #logits = logits / self.CONTRAST_TAU

        ce_criterion = nn.CrossEntropyLoss()
        loss = ce_criterion(logits, labels)
        '''
        mask = (labels != self.IGNORE_LABEL)
        # 去除待忽略像素值
        labels = labels[mask]
        features = feat[mask]

        # 得到原型的类别数和通道数
        class_num, feature_num = Proto.size()  # 7*1024

        # 计算当前图像特征的整体原型
        N, A = features.size()                 # 32768*1024
        C = class_num
        
        features = features.view(N, 1, A).expand(N, C, A)
        onehot = torch.zeros(N, C).cuda()
        onehot.scatter_(1, labels.view(-1, 1), 1)
        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)
        features_by_sort = features.mul(NxCxA_onehot)
        Amount_CXA = NxCxA_onehot.sum(0)
        Amount_CXA[Amount_CXA == 0] = 1
        mean = features_by_sort.sum(0) / Amount_CXA   # mean为当前图像的整体类别原型,7/1024

        # 进行归一化
        Proto = F.normalize(Proto, p=2, dim=1)
        Current_Proto = F.normalize(mean, p=2, dim=1)

        # 计算矩阵乘法
        logits = Current_Proto.mm(Proto.permute(1, 0).contiguous())

        # 设置标签
        label = torch.tensor([[0], [1.], [2.], [3.], [4], [5.],[ 6.]]).to(logits.device)
        #label = torch.eye(class_num).to(logits.device)
        # 计算loss，对图像缺失的类需要置零
        ce_criterion = nn.CrossEntropyLoss()
        loss = 0
        num = 0
        for i in range (class_num):
            if sum(logits[i]) != 0:
                loss = ce_criterion(logits[i].reshape(1,class_num), label[i].long())+loss
                num = num+1

        # 计算loss
        loss  =  loss / max(num,1)


        
        return loss,mean
