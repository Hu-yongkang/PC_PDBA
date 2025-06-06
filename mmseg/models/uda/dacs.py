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

from .projection import ProjectionHead        
from .module_helper import ModuleHelper       
from ..discri.grad_reverse import grad_reverse   
from ..discri.MLP import MLP
from ...core.utils.loss import PrototypeContrastiveLoss  
from ...core.utils.lovasz_loss import lovasz_softmax     
from ...core.utils.prototype_dist_estimator import prototype_dist_estimator   
 

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
        self.local_iter = 0                
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

        
        self.out_dim = 256
        self.proj_final_dim = 128
        self.project = nn.Sequential(
            nn.Conv2d(self.out_dim, self.out_dim, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_dim, self.proj_final_dim, kernel_size=1, stride=1)
        )

        self.classifier = nn.Conv2d(self.out_dim, 6, kernel_size=1, stride=1)      #   

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
        
        self.feat_estimator = prototype_dist_estimator(feature_num= con_feature_num) #  
        self.out_estimator = prototype_dist_estimator(feature_num=self.num_classes ) #  
        LEARNING_RATE_D = 2.5e-4                 
        feat_dim = 1024
        self.Proto_D = MLP(feat_dim, feat_dim, 1, 3)  
 

    def concat_all_gather(self, tensor):
        """
 
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




    def loss_proto_da(self,outputs):
        output_protos = outputs["da_protos"]
        assert output_protos.shape[0] % 2 == 0 
        class_map_source = outputs['class_map_source'] # 
        class_map_target = outputs['class_map_target'] # 
        num_feat         = outputs['num_feat']

        class_num = class_map_source.shape[0]          #  
        targets = torch.empty_like(output_protos)      #  
        targets[:class_num] = 0                        #  
        targets[class_num:] = 1                        #  
        # 
        loss = F.binary_cross_entropy_with_logits(output_protos, targets, reduction='none') # 
        # 
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

 
        
        MULTI_LEVEL= False
        LAMBDA_LOV = 0
        LAMBDA_FEAT: 1.0
        LAMBDA_OUT: 1.0
        DELTA = 0.8 # 
        


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
        src_feat_pop = clean_losses.pop('features')   #  
        clean_loss, clean_log_vars = self._parse_losses(clean_losses)   # 
        print('clean_loss',clean_loss)
        clean_loss = clean_loss*10 # 
        # clean_loss=1.9376   debug  1w
        if self.local_iter < 8000:             #    
            log_vars.update(clean_log_vars)
            clean_loss.backward(retain_graph=self.enable_fdist) #  
        else:
#     --------------- ----------------------- 
        
# 
            src_size = img.shape[-2:]            #  
            src_feat = self.get_model().extract_feat(img)               
             
            src_out, _ = self.model.decode_head.forward(src_feat)   
            tgt_feat = self.get_model().extract_feat(target_img)        
            tgt_out, _ = self.model.decode_head.forward(tgt_feat) #  
   
            _, _, h, w = src_feat[0].size()
            feat1 = src_feat[0]                     #  
            feat2 = F.interpolate(src_feat[1], size=(h, w), mode="bilinear", align_corners=True)
            feat3 = F.interpolate(src_feat[2], size=(h, w), mode="bilinear", align_corners=True)
            feat4 = F.interpolate(src_feat[3], size=(h, w), mode="bilinear", align_corners=True)
            src_feat = torch.cat([feat1, feat2, feat3, feat4], 1)
            src_feat=self.proj_head(src_feat )  #  
             
            
            _, _, h, w = tgt_feat[0].size()
            feat1 = tgt_feat[0]                     #  
            feat2 = F.interpolate(tgt_feat[1], size=(h, w), mode="bilinear", align_corners=True)
            feat3 = F.interpolate(tgt_feat[2], size=(h, w), mode="bilinear", align_corners=True)
            feat4 = F.interpolate(tgt_feat[3], size=(h, w), mode="bilinear", align_corners=True)
            tgt_feat = torch.cat([feat1, feat2, feat3, feat4], 1)
            tgt_feat=self.proj_head(tgt_feat )  #  
             

 
            ce_criterion = nn.CrossEntropyLoss(ignore_index=255) #  
            pcl_criterion = PrototypeContrastiveLoss( )         #  
            src_pred = F.interpolate(src_out, size=src_size, mode='bilinear', align_corners=True)  #  
            if LAMBDA_LOV > 0:              # 
                pred_softmax = F.softmax(src_pred, dim=1)
                loss_lov = lovasz_softmax(pred_softmax, gt_semantic_seg, ignore=255 )   #  
                loss_sup = clean_loss +  LAMBDA_LOV * loss_lov   #  
                #meters.update(loss_lov=loss_lov.item())
            else:
                loss_sup = clean_loss
            #meters.update(loss_sup=loss_sup.item())

             
            B, A, Hs, Ws = src_feat.size()     
            src_mask = F.interpolate(gt_semantic_seg.float(), size=(Hs, Ws), mode='nearest').long()   
            src_mask = src_mask.contiguous().view(B * Hs * Ws, )      
            assert not src_mask.requires_grad
            
            _, _, Ht, Wt = tgt_feat.size()      
            tgt_out0 = tgt_out.clone()
            tgt_out1 = F.softmax(tgt_out0)     
            tgt_out_maxvalue, tgt_mask = torch.max(tgt_out1, dim=1)   
            for i in range(self.num_classes): 
                tgt_mask[(tgt_out_maxvalue <  DELTA) * (tgt_mask == i)] = 255   
            tgt_mask = tgt_mask.contiguous().view(B * Ht * Wt, )        
            assert not tgt_mask.requires_grad

            src_feat1 = src_feat.clone()
            tgt_feat1 = tgt_feat.clone()
            src_feat = src_feat.permute(0, 2, 3, 1).contiguous().view(B * Hs * Ws, A)    
            tgt_feat = tgt_feat.permute(0, 2, 3, 1).contiguous().view(B * Ht * Wt, A)    
             
            
            self.feat_estimator.update(features=src_feat.detach(), labels=src_mask)       
            
            scr_feat_loss, class_prototypes_source = pcl_criterion(Proto=self.feat_estimator.Proto.detach(),
                                        srcfeat= src_feat,       
                                        tarfeat= tgt_feat1,
                                        tarout = tgt_out0,
                                        labels=src_mask)     
            if math.isnan(scr_feat_loss):
                loss_feat = 1.94        
            else:
                loss_feat = scr_feat_loss
 



            #meters.update(loss_feat=loss_feat.item())  

            if MULTI_LEVEL:  # True
                src_out = src_out.permute(0, 2, 3, 1).contiguous().view(B * Hs * Ws, self.num_classes)  #  
                tgt_out = tgt_out.permute(0, 2, 3, 1).contiguous().view(B * Ht * Wt, self.num_classes)  #  

                # update output-level statistics
                
                self.out_estimator.update(features=src_out.detach(), labels=src_mask)
                self.out_estimator.update(features=tgt_out.detach(), labels=tgt_mask)

               
                src_out_loss,_ = pcl_criterion(Proto=self.out_estimator.Proto.detach(),
                                            feat=src_out,
                                            labels=src_mask)
                
                tgt_out_loss,_ = pcl_criterion(Proto=self.out_estimator.Proto.detach(),
                                            feat=tgt_out,
                                            labels=tgt_mask)
                print('Contract_scr_out_loss:',src_out_loss)
                print('Contract_tgt_out_loss:',tgt_out_loss)
                if math.isnan( tgt_out_loss):
                    loss_out = src_out_loss 
                else:
                    loss_out = src_out_loss + tgt_out_loss
 
            else:
                loss_con_all = loss_sup +  1* loss_feat

        
            print('loss_sup',loss_sup)
            print('loss_con_all',loss_con_all)
            loss_con_all.backward(retain_graph=self.enable_fdist) # 
  
        if self.print_grad_magnitude:
            params = self.get_model().backbone.parameters()
            seg_grads = [
                p.grad.detach().clone() for p in params if p.grad is not None
            ]
            grad_mag = calc_grad_magnitude(seg_grads)
            mmcv.print_log(f'Seg. Grad.: {grad_mag}', 'mmseg')

        # ImageNet feature distance
        if self.enable_fdist:   #  
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

        
        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False
        ema_logits, fuse_x = self.get_ema_model().encode_decode(   
            target_img, target_img_metas)

        ema_softmax = torch.softmax(ema_logits.detach(), dim=1)    # 
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)  #  
        ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
         
        ps_size = np.size(np.array(pseudo_label.cpu()))
        pseudo_weight = torch.sum(ps_large_p).item() / ps_size
        pseudo_weight1 = pseudo_weight                             
        pseudo_weight = pseudo_weight * torch.ones(
            pseudo_prob.shape, device=dev)

        if self.psweight_ignore_top > 0:
             
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
         
        stage = 2                        
        mix_losses = self.get_model().forward_train(stage,
            mixed_img, img_metas, mixed_lbl, pseudo_weight, return_feat=True)
        mix_losses.pop('features')
        mix_losses = add_prefix(mix_losses, 'mix')
        mix_loss, mix_log_vars = self._parse_losses(mix_losses)
        print('mix_clean_loss',mix_loss)

        mix_loss = mix_loss * 5  

        log_vars.update(mix_log_vars)   
        if self.local_iter < 0 :   
            mix_loss.backward()
        else:                          
            
            cl=1   
            mix_p_losses = self.get_model().forward_train(cl, mixed_img, target_img_metas, mixed_lbl,pseudo_weight, return_feat=False,p_label=True)
            mix_p_loss, mix_p_loss_log_vars = self._parse_losses(mix_p_losses)
            print('mix_p_loss',mix_p_loss)
            mix_loss = mix_loss+ mix_p_loss            
            print('pseudo_weight1:',pseudo_weight1) # 
            if pseudo_weight1 > 0.55 :                    
  
                pseudo_weight1=torch.tensor(pseudo_weight1)
                p_losses = self.get_model().forward_train(cl, target_img, target_img_metas, pseudo_label,pseudo_weight1, return_feat=False,p_label=True)
                 
                p_loss, p_loss_log_vars = self._parse_losses(p_losses)
                 
                print('p_loss',p_loss)                
                
                mix_loss = mix_loss+p_loss  
       

            mix_loss.backward()

        
        
        
        
        
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
