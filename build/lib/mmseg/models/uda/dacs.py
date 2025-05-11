# The ema model and the domain-mixing are based on:
# https://github.com/vikolss/DACS 

import math
import os
import random
from copy import deepcopy
 
import dataloaders
import json
from itertools import cycle

import torch.nn.functional as F
from torch import nn

import mmcv
import numpy as np
import torch
from matplotlib import pyplot as plt
from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd

from mmseg.core import add_prefix
from mmseg.models import UDA, build_segmentor
from mmseg.models.uda.uda_decorator import UDADecorator, get_module
from mmseg.models.utils.dacs_transforms import (denorm, get_class_masks,
                                                get_mean_std, strong_transform)
from mmseg.models.utils.visualization import subplotimg
from mmseg.utils.utils import downscale_label_ratio

from .projection import ProjectionHead       # 此处新增-------------------------------
from .module_helper import ModuleHelper      # 此处新增---------------------------------
from ..discri.grad_reverse import grad_reverse  # 此处新增------------------------------
from ..discri.MLP import MLP
from ...core.utils.loss import PrototypeContrastiveLoss # 新增----
from ...core.utils.lovasz_loss import lovasz_softmax    # 新增----
from ...core.utils.prototype_dist_estimator import prototype_dist_estimator  # 新增----
 

def _params_equal(ema_model, model):
    for ema_param, param in zip(ema_model.named_parameters(),
                                model.named_parameters()):
        if not torch.equal(ema_param[1].data, param[1].data):
            # print("Difference in", ema_param[0])
            return False
    return True


def calc_grad_magnitude(grads, norm_type=2.0):
    norm_type = float(norm_type)
    if norm_type == math.inf:
        norm = max(p.abs().max() for p in grads)
    else:
        norm = torch.norm(
            torch.stack([torch.norm(p, norm_type) for p in grads]), norm_type)

    return norm


@UDA.register_module()
class DACS(UDADecorator):

    def __init__(self, **cfg):
        super(DACS, self).__init__(**cfg)
        self.local_iter = 0               #  注意对应到读取的模型的iteration
        self.max_iters = cfg['max_iters']
        self.alpha = cfg['alpha']
        self.pseudo_threshold = cfg['pseudo_threshold']
        self.psweight_ignore_top = cfg['pseudo_weight_ignore_top']
        self.psweight_ignore_bottom = cfg['pseudo_weight_ignore_bottom']
        self.fdist_lambda = cfg['imnet_feature_dist_lambda']
        self.fdist_classes = cfg['imnet_feature_dist_classes']
        self.fdist_scale_min_ratio = cfg['imnet_feature_dist_scale_min_ratio']
        self.enable_fdist = self.fdist_lambda > 0
        self.mix = cfg['mix']
        self.blur = cfg['blur']
        self.color_jitter_s = cfg['color_jitter_strength']
        self.color_jitter_p = cfg['color_jitter_probability']
        self.debug_img_interval = cfg['debug_img_interval']
        self.print_grad_magnitude = cfg['print_grad_magnitude']
        assert self.mix == 'class'

        self.debug_fdist_mask = None
        self.debug_gt_rescale = None

        self.class_probs = {}
        ema_cfg = deepcopy(cfg['model'])
        self.ema_model = build_segmentor(ema_cfg)
        self.proj_head = ProjectionHead(dim_in=1024, proj_dim=1024,proj='convmlp') # 新增----

        if self.enable_fdist:
            self.imnet_model = build_segmentor(deepcopy(cfg['model']))
        else:
            self.imnet_model = None

        # CAC ---------------------新增-----------------------
        self.out_dim = 256
        self.proj_final_dim = 128
        self.project = nn.Sequential(
            nn.Conv2d(self.out_dim, self.out_dim, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_dim, self.proj_final_dim, kernel_size=1, stride=1)
        )

        self.classifier = nn.Conv2d(self.out_dim, 6, kernel_size=1, stride=1)      #   注意修改

        self.weight_unsup = 0.1
        self.temp = 0.1
        self.epoch_start_unsup = 0
        self.selected_num = 1600          # ？
        self.step_save = 2
        self.step_count = 0
        self.feature_bank = []
        self.pseudo_label_bank = []
        self.pos_thresh_value = 0.9
        self.stride = 8
        con_feature_num = 1024   #   y原为1024,f3是512
        
        self.feat_estimator = prototype_dist_estimator(feature_num= con_feature_num) # 原型距离估计器 1024，将cfg初始化信息写在模型函数里面，可进入修改
            
        self.out_estimator = prototype_dist_estimator(feature_num=self.num_classes ) # 最终分类数 
        LEARNING_RATE_D = 2.5e-4                 # 新增鉴别器初始学习率
        feat_dim = 1024
        self.Proto_D = MLP(feat_dim, feat_dim, 1, 3) # 新增特征原型鉴别器,hidden_dim为特征维度C
 

    def concat_all_gather(self, tensor):
        """
        Performs all_gather operation on the provided tensors. 对提供的张量执行all_gather操作
        *** Warning ***: torch.distributed.all_gather has no gradient.   无梯度
        """
        with torch.no_grad():
            tensors_gather = [torch.ones_like(tensor)
                for _ in range(1)]
            tensors_gather = [tensor]

            output = torch.cat(tensors_gather, dim=0)
        return output  # ------------------------------------------------------

    def get_ema_model(self):
        return get_module(self.ema_model)

    def get_imnet_model(self):
        return get_module(self.imnet_model)

    def _init_ema_weights(self):
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(self.get_model().parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.get_ema_model().parameters(),
                                    self.get_model().parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]


    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, 
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """

        optimizer.zero_grad()
        log_vars = self(**data_batch)
        optimizer.step()

        log_vars.pop('loss', None)  # remove the unnecessary 'loss'
        outputs = dict(
            log_vars=log_vars, num_samples=len(data_batch['img_metas']))
        return outputs

    def masked_feat_dist(self, f1, f2, mask=None):
        feat_diff = f1 - f2
        # mmcv.print_log(f'fdiff: {feat_diff.shape}', 'mmseg')
        pw_feat_dist = torch.norm(feat_diff, dim=1, p=2)
        # mmcv.print_log(f'pw_fdist: {pw_feat_dist.shape}', 'mmseg')
        if mask is not None:
            # mmcv.print_log(f'fd mask: {mask.shape}', 'mmseg')
            pw_feat_dist = pw_feat_dist[mask.squeeze(1)]
            # mmcv.print_log(f'fd masked: {pw_feat_dist.shape}', 'mmseg')
        return torch.mean(pw_feat_dist)

    def calc_feat_dist(self, img, gt, feat=None):
        assert self.enable_fdist
        with torch.no_grad():
            self.get_imnet_model().eval()
            feat_imnet = self.get_imnet_model().extract_feat(img)
            feat_imnet = [f.detach() for f in feat_imnet]
        lay = -1
        if self.fdist_classes is not None:
            fdclasses = torch.tensor(self.fdist_classes, device=gt.device)
            scale_factor = gt.shape[-1] // feat[lay].shape[-1]
            gt_rescaled = downscale_label_ratio(gt, scale_factor,
                                                self.fdist_scale_min_ratio,
                                                self.num_classes,
                                                255).long().detach()
            fdist_mask = torch.any(gt_rescaled[..., None] == fdclasses, -1)
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay],
                                              fdist_mask)
            self.debug_fdist_mask = fdist_mask
            self.debug_gt_rescale = gt_rescaled
        else:
            feat_dist = self.masked_feat_dist(feat[lay], feat_imnet[lay])
        feat_dist = self.fdist_lambda * feat_dist
        feat_loss, feat_log = self._parse_losses(
            {'loss_imnet_feat_dist': feat_dist})
        feat_log.pop('loss', None)
        return feat_loss, feat_log

    '''
    # 学习率倍数变化算法
    def lr_poly(base_lr, iter, max_iter, power):
        return base_lr * ((1 - float(iter) / max_iter) ** (power))

    # 新增鉴别器的学习率变化                
    def adjust_learning_rate_D(optimizer, i_iter):
        lr = lr_poly (LEARNING_RATE_D, i_iter, 50000, 0.9)
        optimizer.param_groups[0]['lr'] = lr
        if len(optimizer.param_groups) > 1:
            optimizer.param_groups[1]['lr'] = lr * 10
    '''
    def loss_proto_da(self,outputs):
        output_protos = outputs["da_protos"]
        assert output_protos.shape[0] % 2 == 0 
        class_map_source = outputs['class_map_source'] #[7,1] 源域原型的mask，缺类别则置零
        class_map_target = outputs['class_map_target'] #[7,1] 目标域原型的mask，缺类别则置零
        num_feat         = outputs['num_feat']

        class_num = class_map_source.shape[0]          # 类别数为7
        targets = torch.empty_like(output_protos)      # 空矩阵 14/1
        targets[:class_num] = 0                        # 前7 置0
        targets[class_num:] = 1                        # 前7 置1
        #计票损失
        loss = F.binary_cross_entropy_with_logits(output_protos, targets, reduction='none') #[14) 鉴别器损失
        #得到class map
        class_map = torch.cat([class_map_source,class_map_target],dim=0).unsqueeze(1).to(loss.device)
 
        loss = loss * class_map
        loss = loss.sum()/(max(1,num_feat))
        return loss 

    def forward_train(self, img, img_metas, gt_semantic_seg, target_img,
                      target_img_metas):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        log_vars = {}
        batch_size = img.shape[0]
        dev = img.device

        # 用于特征原型对应通道数 -----
        feature_num = 256
        
        MULTI_LEVEL= False
        LAMBDA_LOV = 0
        LAMBDA_FEAT: 1.0
        LAMBDA_OUT: 1.0
        DELTA = 0.8 # ------新增
        


        # Init/update ema model
        if self.local_iter == 0:
            self._init_ema_weights()
            # assert _params_equal(self.get_ema_model(), self.get_model())
        #print('feat_estimator.Proto:',self.feat_estimator.Proto)
        #print('out_estimator.Proto:',self.out_estimator.Proto)
        print("------------------------new_image_in----------------------------------")

        if self.local_iter > 0:
            self._update_ema(self.local_iter)
            # assert not _params_equal(self.get_ema_model(), self.get_model())
            # assert self.get_ema_model().training

        means, stds = get_mean_std(img_metas, dev)
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0)
        }
        # Source stage: 1
        stage = 1
        clean_losses = self.get_model().forward_train(
            stage, img, img_metas, gt_semantic_seg, return_feat=True)
        src_feat_pop = clean_losses.pop('features')   # src_fea = x = [f1,f2,f3,f4]
        clean_loss, clean_log_vars = self._parse_losses(clean_losses)   # 此处损失由源域分割+对比损失，改动可加入伪标签的计算结果
        print('clean_loss',clean_loss)
        clean_loss = clean_loss*10 # 注意---------------------------------------------------
        # clean_loss=1.9376   debug  1w
        if self.local_iter < 0:             #   整体训练是<20000,预加载训练是0（根据要求更改） 
            log_vars.update(clean_log_vars)
            clean_loss.backward(retain_graph=self.enable_fdist) # 最后出现clean_loss 
        else:
#     -----------------此处新增对齐损失------------------------ 
        
# 输入：img, img_metas, gt_semantic_seg, target_img
            src_size = img.shape[-2:]            # 输入尺寸 512 512
            src_feat = self.get_model().extract_feat(img)             # 对  源域 图像捕获特征，len=4,torch.Size([2, 64, 128, 128])， [2, 128, 64, 64])， ([2, 320, 32, 32])  ([2, 512, 16, 16]) 
            # self.model.decode_head.forward 会返回两个值，分类结果和融合后的feature_fuse,因此写作src_out, _ =
            src_out, _ = self.model.decode_head.forward(src_feat)   # 对  源域 特征分类 src_out 修改后为 torch.Size([2, 7, 128, 128])
            tgt_feat = self.get_model().extract_feat(target_img)      # 对  目标域  图像捕获特征，len=4,torch.Size([2, 64, 128, 128])， [2, 128, 64, 64])， ([2, 320, 32, 32])  ([2, 512, 16, 16])
            tgt_out, _ = self.model.decode_head.forward(tgt_feat) # 对  目标域  特征分类tgt_out 修改后为  torch.Size([2, 7, 128, 128])
    # 对特征进行拼接，此处可进行修改，或可改为ASPP
            _, _, h, w = src_feat[0].size()
            feat1 = src_feat[0]                     # 将f2-4 上采样，得到f1大小后沿着C拼接得到feats
            feat2 = F.interpolate(src_feat[1], size=(h, w), mode="bilinear", align_corners=True)
            feat3 = F.interpolate(src_feat[2], size=(h, w), mode="bilinear", align_corners=True)
            feat4 = F.interpolate(src_feat[3], size=(h, w), mode="bilinear", align_corners=True)
            src_feat = torch.cat([feat1, feat2, feat3, feat4], 1)
            src_feat=self.proj_head(src_feat )  # torch.Size([2, 1024, 128, 128]),改动，仅使用第四层
            #src_feat = src_feat[3]
            
            _, _, h, w = tgt_feat[0].size()
            feat1 = tgt_feat[0]                     # 将f2-4 上采样，得到f1大小后沿着C拼接得到feats
            feat2 = F.interpolate(tgt_feat[1], size=(h, w), mode="bilinear", align_corners=True)
            feat3 = F.interpolate(tgt_feat[2], size=(h, w), mode="bilinear", align_corners=True)
            feat4 = F.interpolate(tgt_feat[3], size=(h, w), mode="bilinear", align_corners=True)
            tgt_feat = torch.cat([feat1, feat2, feat3, feat4], 1)
            tgt_feat=self.proj_head(tgt_feat )  # torch.Size([2, 1024, 128, 128]) 改动，仅使用第四层
            #tgt_feat = tgt_feat[3]

    # 损失定义
            ce_criterion = nn.CrossEntropyLoss(ignore_index=255) # 改
            pcl_criterion = PrototypeContrastiveLoss( )         # 要输入Proto, feat, labels，注意此处为了方便，将cfg初始化信息写在损失函数里面，可进入修改
    # 特征原型，此处原本构造特征原型，单会导致每一个图像进入后，特征原型初始化，因此该位置错误
            '''
            feat_estimator = prototype_dist_estimator(feature_num=feature_num ) # 原型距离估计器 1024，将cfg初始化信息写在模型函数里面，可进入修改
            if MULTI_LEVEL:    # True
                out_estimator = prototype_dist_estimator(feature_num=self.num_classes ) # 最终分类数        
            '''
    # 有监督损失计算，注意标签要进行处理,src_pred torch.Size([2, 7, 512, 512])
            src_pred = F.interpolate(src_out, size=src_size, mode='bilinear', align_corners=True)  # 待查[]
            if LAMBDA_LOV > 0:              # LAMBDA_LOV: 0.75  ?该loss的意义？
                pred_softmax = F.softmax(src_pred, dim=1)
                loss_lov = lovasz_softmax(pred_softmax, gt_semantic_seg, ignore=255 )   # 参考值0.857源域seg loss，注意标签处理,
                loss_sup = clean_loss +  LAMBDA_LOV * loss_lov   # 注意此处 clean_loss
                #meters.update(loss_lov=loss_lov.item())
            else:
                loss_sup = clean_loss
            #meters.update(loss_sup=loss_sup.item())

            # label下采样
            # source mask: downsample the ground-truth label
            B, A, Hs, Ws = src_feat.size()   # torch.Size([2, 1024, 128, 128])
            src_mask = F.interpolate(gt_semantic_seg.float(), size=(Hs, Ws), mode='nearest').long() # torch.Size([2, 1, 128, 128])
            src_mask = src_mask.contiguous().view(B * Hs * Ws, )     # torch.Size([32768])
            assert not src_mask.requires_grad
            # target mask: constant threshold -- cfg.SOLVER.THRESHOLD   固定阈值THRESHOLD=0.6
            _, _, Ht, Wt = tgt_feat.size()       # 128  128
            tgt_out1 = tgt_out.clone()
            tgt_out1 = F.softmax(tgt_out1)
            tgt_out_maxvalue, tgt_mask = torch.max(tgt_out1, dim=1)  # 最大值位置 Size([2, 128, 128])、目标域mask Size([2, 128, 128])
            for i in range(self.num_classes):# 注意标签是否要对应
                tgt_mask[(tgt_out_maxvalue <  DELTA) * (tgt_mask == i)] = 255  # 预测置信度<0.9则置位255
            tgt_mask = tgt_mask.contiguous().view(B * Ht * Wt, )       # torch.Size([32768])
            assert not tgt_mask.requires_grad

            src_feat = src_feat.permute(0, 2, 3, 1).contiguous().view(B * Hs * Ws, A)   # torch.Size([32768, 1024])
            tgt_feat = tgt_feat.permute(0, 2, 3, 1).contiguous().view(B * Ht * Wt, A)   # torch.Size([32768, 1024])
            # update feature-level statistics  更新特征级统计信息
            
            self.feat_estimator.update(features=src_feat.detach(), labels=src_mask)
            self.feat_estimator.update(features=tgt_feat.detach(), labels=tgt_mask)
    # 判断程序仅限于 目标域的mask，源域loss是正常的。------------------------------------注意方式。
            # contrastive loss on both domains  ,debug情况下loss_feat= nan，feat_estimator.Proto.shape=[7,1024]
            
    # 更改特征-整体原型损失 为 单图原型-整体原型损失 
    # ---------------------------------------------------------------------------   
    
    # ---------------------------------------------------------------------------     
            
            scr_feat_loss, class_prototypes_source = pcl_criterion(Proto=self.feat_estimator.Proto.detach(),
                                        feat=src_feat,      # torch.Size([32768, 1024]),src_mask=torch.Size([32768])
                                        labels=src_mask)
            
            tgt_feat_loss, class_prototypes_target = pcl_criterion(Proto=self.feat_estimator.Proto.detach(),
                                                feat=tgt_feat,
                                                labels=tgt_mask)
            
            print('Contract_scr_feat_loss:',scr_feat_loss)
            print('Contract_tgt_feat_loss:',tgt_feat_loss)

            if math.isnan(tgt_feat_loss):
                loss_feat = scr_feat_loss       # 1.94
            else:
                loss_feat = scr_feat_loss + tgt_feat_loss
            
            # 开始添加鉴别器损失--------------------------------------------
            
            bce_loss = torch.nn.MSELoss()
            # 将S-T原型进行拼接
            class_prototypes = torch.cat([class_prototypes_source,class_prototypes_target],dim=0) # [14,1024]
            
            # 设置鉴别器,得到预测
            class_prototypes_da = self.Proto_D(grad_reverse(class_prototypes)) #[14,1]

            # 设置源域/目标域的类别mask
            a,b = class_prototypes_da.shape
            class_map = torch.zeros([a,b])  # [14,1]
            num_feat = 0
            class_num = class_prototypes_source.shape[0]  # 类别数7
            for i in range(a):
                if sum(class_prototypes[i]) != 0:
                    class_map[i] = 1
                    num_feat = num_feat +1


            # 计算损失
            da_output = {
                            'da_protos': class_prototypes_da,         # 特征原型
                            'class_map_source': class_map[:class_num] ,
                            'class_map_target': class_map[class_num:] ,
                            'num_feat' :num_feat
                        }
            loss_dis_feat = self.loss_proto_da(da_output)

 




            #meters.update(loss_feat=loss_feat.item())  

            if MULTI_LEVEL:  # True
                src_out = src_out.permute(0, 2, 3, 1).contiguous().view(B * Hs * Ws, self.num_classes)  # torch.Size([32768, 7])
                tgt_out = tgt_out.permute(0, 2, 3, 1).contiguous().view(B * Ht * Wt, self.num_classes)  # torch.Size([32768, 7])

                # update output-level statistics
                
                self.out_estimator.update(features=src_out.detach(), labels=src_mask)
                self.out_estimator.update(features=tgt_out.detach(), labels=tgt_mask)

                # the proposed contrastive loss on prediction map    debug情况下loss_feat= nan,out_estimator.Proto.shape=7,7
                src_out_loss,_ = pcl_criterion(Proto=self.out_estimator.Proto.detach(),
                                            feat=src_out,
                                            labels=src_mask)
                
                tgt_out_loss,_ = pcl_criterion(Proto=self.out_estimator.Proto.detach(),
                                            feat=tgt_out,
                                            labels=tgt_mask)
                print('Contract_scr_out_loss:',src_out_loss)
                print('Contract_tgt_out_loss:',tgt_out_loss)
                if math.isnan(tgt_feat_loss):
                    loss_out = src_out_loss 
                else:
                    loss_out = src_out_loss + tgt_out_loss

                '''
                loss_out = pcl_criterion(Proto=out_estimator.Proto.detach(),
                                            feat=src_out,
                                            labels=src_mask) \
                            + pcl_criterion(Proto=out_estimator.Proto.detach(),
                                            feat=tgt_out,
                                            labels=tgt_mask)
                #meters.update(loss_out=loss_out.item())
                '''
                '''
                loss_con_all = (loss_sup \
                        +   loss_feat \
                        +  loss_out )       #      注意损失权重的改动
                '''
 
                print('loss_dis_feat', loss_dis_feat)
#                loss_con_all =loss_con_all  + 0.1*loss_dis_feat
                loss_con_all = loss_sup + 0.1*loss_dis_feat
            else:
                loss_con_all = loss_sup +  0.1* loss_feat

        
            print('loss_sup',loss_sup)
            print('loss_con_all',loss_con_all)
            loss_con_all.backward(retain_graph=self.enable_fdist) # 第一次约3.0，loss_con_all表示监督loss+对抗loss+GT对比loss 

            #optimizer_D.step()         
        

#     --------------------此处结束----------------------------

        #log_vars.update(clean_log_vars)
        # clean_loss.backward(retain_graph=self.enable_fdist) # 最后出现clean_loss
        #loss_con_all.backward(retain_graph=self.enable_fdist) # 第一次约3.0，loss_con_all表示监督loss+对抗loss+GT对比loss 
        if self.print_grad_magnitude:
            params = self.get_model().backbone.parameters()
            seg_grads = [
                p.grad.detach().clone() for p in params if p.grad is not None
            ]
            grad_mag = calc_grad_magnitude(seg_grads)
            mmcv.print_log(f'Seg. Grad.: {grad_mag}', 'mmseg')

        # ImageNet feature distance
        if self.enable_fdist:   # imnet_feature_dist_classes=[2,3,4]
            feat_loss, feat_log = self.calc_feat_dist(img, gt_semantic_seg, 
                                                      src_feat_pop)

            
            print('feat_con_loss',feat_loss)
            feat_loss.backward()
            log_vars.update(add_prefix(feat_log, 'src'))
            if self.print_grad_magnitude:
                params = self.get_model().backbone.parameters()
                fd_grads = [
                    p.grad.detach() for p in params if p.grad is not None
                ]
                fd_grads = [g2 - g1 for g1, g2 in zip(seg_grads, fd_grads)]
                grad_mag = calc_grad_magnitude(fd_grads)
                mmcv.print_log(f'Fdist Grad.: {grad_mag}', 'mmseg')

        # Generate pseudo-label-------------------------------------
        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False
        ema_logits, fuse_x = self.get_ema_model().encode_decode(   # ema_logits是预测map,b/c/h/w
            target_img, target_img_metas)

        ema_softmax = torch.softmax(ema_logits.detach(), dim=1)    # 归一化
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)  # pseudo_prob为每个像素的置信度
        ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
        #ps_large_p = pseudo_prob.ge(0.2).long() == 1    #  此行最后消除---------------------------
        ps_size = np.size(np.array(pseudo_label.cpu()))
        pseudo_weight = torch.sum(ps_large_p).item() / ps_size
        pseudo_weight1 = pseudo_weight                             # 作为是否参与伪标签对比的阈值对比
        pseudo_weight = pseudo_weight * torch.ones(
            pseudo_prob.shape, device=dev)

        if self.psweight_ignore_top > 0:
            # Don't trust pseudo-labels in regions with potential
            # rectification artifacts. This can lead to a pseudo-label
            # drift from sky towards building or traffic light.
            pseudo_weight[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
        gt_pixel_weight = torch.ones((pseudo_weight.shape), device=dev)

        # Apply mixing
        mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
        mix_masks = get_class_masks(gt_semantic_seg)

        for i in range(batch_size):
            strong_parameters['mix'] = mix_masks[i]
            mixed_img[i], mixed_lbl[i] = strong_transform(
                strong_parameters,
                data=torch.stack((img[i], target_img[i])),
                target=torch.stack((gt_semantic_seg[i][0], pseudo_label[i])))
            _, pseudo_weight[i] = strong_transform(
                strong_parameters,
                target=torch.stack((gt_pixel_weight[i], pseudo_weight[i])))
        mixed_img = torch.cat(mixed_img)
        mixed_lbl = torch.cat(mixed_lbl)
        # Tatget (Mix) stage: 2-------------------------------------------------------------------------
        stage = 2                        # pseudo_weight torch.Size([2, 512, 512])
        mix_losses = self.get_model().forward_train(stage,
            mixed_img, img_metas, mixed_lbl, pseudo_weight, return_feat=True)
        mix_losses.pop('features')
        mix_losses = add_prefix(mix_losses, 'mix')
        mix_loss, mix_log_vars = self._parse_losses(mix_losses)
        print('mix_clean_loss',mix_loss)

        mix_loss = mix_loss * 5 # 新增----------------------------

        log_vars.update(mix_log_vars)   
        if self.local_iter < 4000 : # warm up iterations   ----------此行开始有改动，当iter<1w时保持训练不变 #   整体训练是<10000,预加载训练是0（根据要求更改）
            mix_loss.backward()
        else:                           # 当iter大于1w会额外执行如下：
            '新增目标域的对比训练'
            cl=1 # 代替stage=2的标识符，使得计算像素对比损失
            mix_p_losses = self.get_model().forward_train(cl, mixed_img, target_img_metas, mixed_lbl,pseudo_weight, return_feat=False,p_label=True)
            mix_p_loss, mix_p_loss_log_vars = self._parse_losses(mix_p_losses)
            print('mix_p_loss',mix_p_loss)
            mix_loss = mix_loss+ mix_p_loss            
            print('pseudo_weight1:',pseudo_weight1) # (在第1w个iter情况下，多数标签在0.16-0.4)
            if pseudo_weight1 > 0.55 :                    
                
                #print('---------------------------------into p-label-------------------------')
                pseudo_weight1=torch.tensor(pseudo_weight1)
                p_losses = self.get_model().forward_train(cl, target_img, target_img_metas, pseudo_label,pseudo_weight1, return_feat=False,p_label=True)
                '''
                print('---------------------------')
                print('p_losses.shape',len(p_losses))   # 2
                print('p_losses.type',type(p_losses))   # dict
                print('p_losses.keys',p_losses.keys())  # dict_keys([ 'decode.loss_seg', 'decode.acc_seg'])
                '''
                # 将p_loss加入到mix_loss
                p_loss, p_loss_log_vars = self._parse_losses(p_losses)
                '''
                print('p_loss.type',type(p_loss))     # Tensor
                print('p_loss.shape',(p_loss.shape))  # torch.Size([])
                '''
                print('p_loss',p_loss)                # 0.6589,最终为0.1x~0.2x
                
                mix_loss = mix_loss+p_loss #      损失计算要改参数
            '''
            else:
                p_loss = 0
            print('p-lableloss:',p_loss)
            ''' 
            '''
            # DATA_LOADER 2
            def init_dl():
                # LOAD Config
                config = json.load(open('configs/cac/Urban.json'))
                # DATA LOADERS
                config['train_supervised']['n_labeled_examples'] = config['n_labeled_examples'] # 2975
                config['train_unsupervised']['n_labeled_examples'] = config['n_labeled_examples']
                config['train_unsupervised']['use_weak_lables'] = config['use_weak_lables'] # False
                config['train_supervised']['data_dir'] = config['data_dir']   # "data/cityscapes"
                config['train_unsupervised']['data_dir'] = config['data_dir'] # "data/cityscapes"
                config['val_loader']['data_dir'] = config['data_dir']         # "data/cityscapes"
                config['train_supervised']['datalist'] = config['datalist']   # 2
                config['train_unsupervised']['datalist'] = config['datalist'] # 2
                config['val_loader']['datalist'] = config['datalist']         # 2
                if config['dataset'] == 'LoveDAUrban':
                    sup_dataloader = dataloaders.Urban         # 给与目标域的原图和原标签 
                    unsup_dataloader = dataloaders.PairUrban   # 给与两次裁剪后拼接（dim=0）的图像和标签，以及重叠区域在两个裁剪图像上的位置信息+翻转的bool列表

                supervised_loader = sup_dataloader(config['train_supervised'])
                unsupervised_loader = unsup_dataloader(config['train_unsupervised'])
                self.supervised_loader = supervised_loader
                self.unsupervised_loader = unsupervised_loader

                self.dataloader = iter(zip(cycle(self.supervised_loader), cycle(self.unsupervised_loader)))
            a = 1
            if a == 1:
                init_dl()
                a += 1

            # Patch Contrast  补丁的对比
            (input_l, target_l), (input_ul, target_ul, ul1, br1, ul2, br2, flip) = next(self.dataloader)
            input_l, target_l = input_l.cuda(non_blocking=True), target_l.cuda(non_blocking=True)
            input_ul, target_ul = input_ul.cuda(non_blocking=True), target_ul.cuda(non_blocking=True)
            kargs = {'gpu': 0, 'ul1': ul1, 'br1': br1, 'ul2': ul2, 'br2': br2, 'flip': flip}
            # Params into the model
            xl = input_l
            target_l = target_l
            x_ul = input_ul
            target_ul = target_ul
            _, fuse_xl = self.get_ema_model().encode_decode(
                xl, target_img_metas)
            x_ul1 = x_ul[:, 0, :, :, :]     # 取裁剪图像1
            x_ul2 = x_ul[:, 1, :, :, :]     # 取裁剪图像2

            # Process xul1
            _, fuse_xul1 = self.get_ema_model().encode_decode(
                x_ul1, target_img_metas)
            enc_ul1 = fuse_xul1                                       # 裁剪图像1的特征
            # Downsample
            enc_ul1 = F.avg_pool2d(enc_ul1, kernel_size=2, stride=2)  # 下采样
            output_ul1 = self.project(enc_ul1)  # [b, c, h, w]        # 过两个线性层 
            output_ul1 = F.normalize(output_ul1, 2, 1)
            # Process xul2
            _, fuse_xul2 = self.get_ema_model().encode_decode(
                x_ul2, target_img_metas)
            enc_ul2 = fuse_xul2                                       # 裁剪图像2的特征
            # Downsample
            enc_ul2 = F.avg_pool2d(enc_ul2, kernel_size=2, stride=2)
            output_ul2 = self.project(enc_ul2)  # [b, c, h, w]
            output_ul2 = F.normalize(output_ul2, 2, 1)

            # compute pseudo label 
            logits1 = self.classifier(enc_ul1)  # [batch_size, num_classes, h, w] 裁剪图像1预测
            logits2 = self.classifier(enc_ul2)                        # 裁剪图像2预测
            pseudo_logits_1 = F.softmax(logits1, 1).max(1)[0].detach()  # [batch_size, h, w] 裁剪图像1伪标签
            pseudo_logits_2 = F.softmax(logits2, 1).max(1)[0].detach()  # 裁剪图像2伪标签
            pseudo_label1 = logits1.max(1)[1].detach()  # [batch_size, h, w]
            pseudo_label2 = logits2.max(1)[1].detach()

            # get overlap part  得到重叠部分
            output_feature_list1 = []
            output_feature_list2 = []
            pseudo_label_list1 = []
            pseudo_label_list2 = []
            pseudo_logits_list1 = []
            pseudo_logits_list2 = []
            for idx in range(x_ul1.size(0)):
                output_ul1_idx = output_ul1[idx]           # output_ul1为特征层
                output_ul2_idx = output_ul2[idx]
                pseudo_label1_idx = pseudo_label1[idx]
                pseudo_label2_idx = pseudo_label2[idx]
                pseudo_logits_1_idx = pseudo_logits_1[idx]
                pseudo_logits_2_idx = pseudo_logits_2[idx]
                if flip[0][idx] == True:
                    output_ul1_idx = torch.flip(output_ul1_idx, dims=(2,))
                    pseudo_label1_idx = torch.flip(pseudo_label1_idx, dims=(1,))
                    pseudo_logits_1_idx = torch.flip(pseudo_logits_1_idx, dims=(1,))
                if flip[1][idx] == True:
                    output_ul2_idx = torch.flip(output_ul2_idx, dims=(2,))
                    pseudo_label2_idx = torch.flip(pseudo_label2_idx, dims=(1,))
                    pseudo_logits_2_idx = torch.flip(pseudo_logits_2_idx, dims=(1,))
                output_feature_list1.append(
                    output_ul1_idx[:, ul1[0][idx] // 8:br1[0][idx] // 8, ul1[1][idx] // 8:br1[1][idx] // 8].permute(1, 2,
                                                                                                                    0).contiguous().view(
                        -1, output_ul1.size(1)))
                output_feature_list2.append(
                    output_ul2_idx[:, ul2[0][idx] // 8:br2[0][idx] // 8, ul2[1][idx] // 8:br2[1][idx] // 8].permute(1, 2,
                                                                                                                    0).contiguous().view(
                        -1, output_ul2.size(1)))
                pseudo_label_list1.append(pseudo_label1_idx[ul1[0][idx] // 8:br1[0][idx] // 8,
                                          ul1[1][idx] // 8:br1[1][idx] // 8].contiguous().view(-1))
                pseudo_label_list2.append(pseudo_label2_idx[ul2[0][idx] // 8:br2[0][idx] // 8,
                                          ul2[1][idx] // 8:br2[1][idx] // 8].contiguous().view(-1))
                pseudo_logits_list1.append(pseudo_logits_1_idx[ul1[0][idx] // 8:br1[0][idx] // 8,
                                           ul1[1][idx] // 8:br1[1][idx] // 8].contiguous().view(-1))
                pseudo_logits_list2.append(pseudo_logits_2_idx[ul2[0][idx] // 8:br2[0][idx] // 8,
                                           ul2[1][idx] // 8:br2[1][idx] // 8].contiguous().view(-1))
            output_feat1 = torch.cat(output_feature_list1, 0)  # [n, c]
            output_feat2 = torch.cat(output_feature_list2, 0)  # [n, c]
            pseudo_label1_overlap = torch.cat(pseudo_label_list1, 0)  # [n,]
            pseudo_label2_overlap = torch.cat(pseudo_label_list2, 0)  # [n,]
            pseudo_logits1_overlap = torch.cat(pseudo_logits_list1, 0)  # [n,]
            pseudo_logits2_overlap = torch.cat(pseudo_logits_list2, 0)  # [n,]
            assert output_feat1.size(0) == output_feat2.size(0)
            assert pseudo_label1_overlap.size(0) == pseudo_label2_overlap.size(0)
            assert output_feat1.size(0) == pseudo_label1_overlap.size(0)

            # concat across multi-gpus  跨多个gpu连接
            b, c, h, w = output_ul1.size()
            selected_num = self.selected_num  # 1600
            output_ul1_flatten = output_ul1.permute(0, 2, 3, 1).contiguous().view(b * h * w, c)
            output_ul2_flatten = output_ul2.permute(0, 2, 3, 1).contiguous().view(b * h * w, c)
            selected_idx1 = np.random.choice(range(b * h * w), selected_num, replace=False) # 抽1600随机数
            selected_idx2 = np.random.choice(range(b * h * w), selected_num, replace=False)
            output_ul1_flatten_selected = output_ul1_flatten[selected_idx1]
            output_ul2_flatten_selected = output_ul2_flatten[selected_idx2]
            output_ul_flatten_selected = torch.cat([output_ul1_flatten_selected, output_ul2_flatten_selected],
                                                   0)  # [2*kk, c]

            output_ul_all = self.concat_all_gather(output_ul_flatten_selected)

            pseudo_label1_flatten_selected = pseudo_label1.view(-1)[selected_idx1]
            pseudo_label2_flatten_selected = pseudo_label2.view(-1)[selected_idx2]
            pseudo_label_flatten_selected = torch.cat([pseudo_label1_flatten_selected, pseudo_label2_flatten_selected],
                                                      0)  # [2*kk]

            pseudo_label_all = self.concat_all_gather(pseudo_label_flatten_selected)

            self.feature_bank.append(output_ul_all)
            self.pseudo_label_bank.append(pseudo_label_all)
            if self.step_count > self.step_save:
                self.feature_bank = self.feature_bank[1:]
                self.pseudo_label_bank = self.pseudo_label_bank[1:]
            else:
                self.step_count += 1
            output_ul_all = torch.cat(self.feature_bank, 0)
            pseudo_label_all = torch.cat(self.pseudo_label_bank, 0)
            eps = 1e-8
            pos1 = (output_feat1 * output_feat2.detach()).sum(-1, keepdim=True) / self.temp  # [n, 1]
            pos2 = (output_feat1.detach() * output_feat2).sum(-1, keepdim=True) / self.temp  # [n, 1]

            # compute loss1
            b = 8000

            def run1(pos, output_feat1, output_ul_idx, pseudo_label_idx, pseudo_label1_overlap, neg_max1):
                # print("gpu: {}, i_1: {}".format(gpu, i))
                mask1_idx = (pseudo_label_idx.unsqueeze(0) != pseudo_label1_overlap.unsqueeze(-1)).float()  # [n, b]
                neg1_idx = (output_feat1 @ output_ul_idx.T) / self.temp  # [n, b]
                logits1_neg_idx = (torch.exp(neg1_idx - neg_max1) * mask1_idx).sum(-1)  # [n, ]
                return logits1_neg_idx

            def run1_0(pos, output_feat1, output_ul_idx, pseudo_label_idx, pseudo_label1_overlap):
                # print("gpu: {}, i_1_0: {}".format(gpu, i))
                mask1_idx = (pseudo_label_idx.unsqueeze(0) != pseudo_label1_overlap.unsqueeze(-1)).float()  # [n, b]
                neg1_idx = (output_feat1 @ output_ul_idx.T) / self.temp  # [n, b]
                neg1_idx = torch.cat([pos, neg1_idx], 1)  # [n, 1+b]
                mask1_idx = torch.cat([torch.ones(mask1_idx.size(0), 1).float().cuda(), mask1_idx], 1)  # [n, 1+b]
                neg_max1 = torch.max(neg1_idx, 1, keepdim=True)[0]  # [n, 1]
                logits1_neg_idx = (torch.exp(neg1_idx - neg_max1) * mask1_idx).sum(-1)  # [n, ]
                return logits1_neg_idx, neg_max1

            N = output_ul_all.size(0)
            logits1_down = torch.zeros(pos1.size(0)).float().cuda()
            for i in range((N - 1) // b + 1):
                # print("gpu: {}, i: {}".format(gpu, i))
                pseudo_label_idx = pseudo_label_all[i * b:(i + 1) * b]
                output_ul_idx = output_ul_all[i * b:(i + 1) * b]
                if i == 0:
                    logits1_neg_idx, neg_max1 = torch.utils.checkpoint.checkpoint(run1_0, pos1, output_feat1, output_ul_idx,
                                                                                  pseudo_label_idx, pseudo_label1_overlap)
                else:
                    logits1_neg_idx = torch.utils.checkpoint.checkpoint(run1, pos1, output_feat1, output_ul_idx,
                                                                        pseudo_label_idx, pseudo_label1_overlap, neg_max1)
                logits1_down += logits1_neg_idx

            logits1 = torch.exp(pos1 - neg_max1).squeeze(-1) / (logits1_down + eps)

            pos_mask_1 = ((pseudo_logits2_overlap > self.pos_thresh_value) & (
                    pseudo_logits1_overlap < pseudo_logits2_overlap)).float()
            loss1 = -torch.log(logits1 + eps)
            loss1 = (loss1 * pos_mask_1).sum() / (pos_mask_1.sum() + 1e-12)

            # compute loss2
            def run2(pos, output_feat2, output_ul_idx, pseudo_label_idx, pseudo_label2_overlap, neg_max2):
                # print("gpu: {}, i_2: {}".format(gpu, i))
                mask2_idx = (pseudo_label_idx.unsqueeze(0) != pseudo_label2_overlap.unsqueeze(-1)).float()  # [n, b]
                neg2_idx = (output_feat2 @ output_ul_idx.T) / self.temp  # [n, b]
                logits2_neg_idx = (torch.exp(neg2_idx - neg_max2) * mask2_idx).sum(-1)  # [n, ]
                return logits2_neg_idx

            def run2_0(pos, output_feat2, output_ul_idx, pseudo_label_idx, pseudo_label2_overlap):
                # print("gpu: {}, i_2_0: {}".format(gpu, i))
                mask2_idx = (pseudo_label_idx.unsqueeze(0) != pseudo_label2_overlap.unsqueeze(-1)).float()  # [n, b]
                neg2_idx = (output_feat2 @ output_ul_idx.T) / self.temp  # [n, b]
                neg2_idx = torch.cat([pos, neg2_idx], 1)  # [n, 1+b]
                mask2_idx = torch.cat([torch.ones(mask2_idx.size(0), 1).float().cuda(), mask2_idx], 1)  # [n, 1+b]
                neg_max2 = torch.max(neg2_idx, 1, keepdim=True)[0]  # [n, 1]
                logits2_neg_idx = (torch.exp(neg2_idx - neg_max2) * mask2_idx).sum(-1)  # [n, ]
                return logits2_neg_idx, neg_max2

            N = output_ul_all.size(0)
            logits2_down = torch.zeros(pos2.size(0)).float().cuda()
            for i in range((N - 1) // b + 1):
                pseudo_label_idx = pseudo_label_all[i * b:(i + 1) * b]
                output_ul_idx = output_ul_all[i * b:(i + 1) * b]
                if i == 0:
                    logits2_neg_idx, neg_max2 = torch.utils.checkpoint.checkpoint(run2_0, pos2, output_feat2, output_ul_idx,
                                                                                  pseudo_label_idx, pseudo_label2_overlap)
                else:
                    logits2_neg_idx = torch.utils.checkpoint.checkpoint(run2, pos2, output_feat2, output_ul_idx,
                                                                        pseudo_label_idx, pseudo_label2_overlap, neg_max2)
                logits2_down += logits2_neg_idx

            logits2 = torch.exp(pos2 - neg_max2).squeeze(-1) / (logits2_down + eps)

            pos_mask_2 = ((pseudo_logits1_overlap > self.pos_thresh_value) & (
                    pseudo_logits2_overlap < pseudo_logits1_overlap)).float()

            loss2 = -torch.log(logits2 + eps)
            loss2 = (loss2 * pos_mask_2).sum() / (pos_mask_2.sum() + 1e-12)
            loss_unsup = self.weight_unsup * (loss1 + loss2)    # α = β = 0.1
            mix_loss = mix_loss + loss_unsup           #  改-------------------------
            '''
            mix_loss.backward()

        # -----------------------------此行以下修改结束---------------------------------
        
        
        
        
        if self.local_iter % self.debug_img_interval == 0:
            out_dir = os.path.join(self.train_cfg['work_dir'],
                                   'class_mix_debug')
            os.makedirs(out_dir, exist_ok=True)
            vis_img = torch.clamp(denorm(img, means, stds), 0, 1)
            vis_trg_img = torch.clamp(denorm(target_img, means, stds), 0, 1)
            vis_mixed_img = torch.clamp(denorm(mixed_img, means, stds), 0, 1)
            for j in range(batch_size):
                rows, cols = 2, 5
                fig, axs = plt.subplots(
                    rows,
                    cols,
                    figsize=(3 * cols, 3 * rows),
                    gridspec_kw={
                        'hspace': 0.1,
                        'wspace': 0,
                        'top': 0.95,
                        'bottom': 0,
                        'right': 1,
                        'left': 0
                    },
                )
                subplotimg(axs[0][0], vis_img[j], 'Source Image')
                subplotimg(axs[1][0], vis_trg_img[j], 'Target Image')
                subplotimg(
                    axs[0][1],
                    gt_semantic_seg[j],
                    'Source Seg GT',
                    cmap='cityscapes')
                subplotimg(
                    axs[1][1],
                    pseudo_label[j],
                    'Target Seg (Pseudo) GT',
                    cmap='cityscapes')
                subplotimg(axs[0][2], vis_mixed_img[j], 'Mixed Image')
                subplotimg(
                    axs[1][2], mix_masks[j][0], 'Domain Mask', cmap='gray')
                # subplotimg(axs[0][3], pred_u_s[j], "Seg Pred",
                #            cmap="cityscapes")
                subplotimg(
                    axs[1][3], mixed_lbl[j], 'Seg Targ', cmap='cityscapes')
                subplotimg(
                    axs[0][3], pseudo_weight[j], 'Pseudo W.', vmin=0, vmax=1)
                if self.debug_fdist_mask is not None:
                    subplotimg(
                        axs[0][4],
                        self.debug_fdist_mask[j][0],
                        'FDist Mask',
                        cmap='gray')
                if self.debug_gt_rescale is not None:
                    subplotimg(
                        axs[1][4],
                        self.debug_gt_rescale[j],
                        'Scaled GT',
                        cmap='cityscapes')
                for ax in axs.flat:
                    ax.axis('off')
                plt.savefig(
                    os.path.join(out_dir,
                                 f'{(self.local_iter + 1):06d}_{j}.png'))
                plt.close()
        self.local_iter += 1

        return log_vars
