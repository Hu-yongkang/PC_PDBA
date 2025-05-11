import torch
import torch.nn as nn
import torch.nn.functional as F



class PrototypeContrastiveLoss(nn.Module):  
    def __init__(self):
        super(PrototypeContrastiveLoss, self).__init__()
        #self.cfg = cfg
        self.IGNORE_LABEL = 255
        self.CONTRAST_TAU = 50.0

    def forward_loss(self, a, b):
        a = F.softmax(a, dim=1)
        b = 1 - b
        out = torch.sum(torch.mul(a, b), dim=1)
        out = out.mean()
        return out

    def reverse_loss(self, a, b):
        n, c, h, w = a.size()
        a = a.permute(1, 0, 2, 3).contiguous().view(c, -1)
        b = b.permute(1, 0, 2, 3).contiguous().view(c, -1)
        a = F.softmax(a, dim=1)
        b = 1.0 - b
        out = torch.sum(torch.mul(a, b), dim=1)
        out = out.mean()
        return out

    def contrast_cal(self, x1, x2):
        temperature = 1
        n, c, h, w = x1.shape
        X_ = F.normalize(x1, p=2, dim=1)
        X_ = X_.permute(0, 2, 3, 1).contiguous().view(-1, c) # nhw,c
        Y_ = x2.contiguous().view(6, 1024)  # class,c
        Y_ = F.normalize(Y_, p=2, dim=-1)
        out = torch.div(torch.matmul(X_, Y_.T), temperature) # nhw,class
        out = out.contiguous().view(n, h, w, 6).permute(0, 3, 1, 2) # n,class,h,w
        return out
    
    
    def forward(self, Proto, srcfeat,tarfeat, tarout,labels): #原型，目标域特征，目标域输出，
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
        assert srcfeat.requires_grad
        assert srcfeat.dim() == 2
        assert labels.dim() == 1
        
        mask = (labels != self.IGNORE_LABEL)
        # 去除待忽略像素值
        labels = labels[mask]
        features = srcfeat[mask]

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



        target_contrast = self.contrast_cal(tarfeat, Proto)
        loss_contrast1 = self.forward_loss(tarout, target_contrast)
        loss_contrast2 = self.reverse_loss(tarout, target_contrast)
        loss_contrast = loss_contrast1+loss_contrast2
        #print("forward_loss:",loss_contrast1)
        #print("reverse_loss:",loss_contrast2)
        
        return loss_contrast,mean
       
        

 
        

