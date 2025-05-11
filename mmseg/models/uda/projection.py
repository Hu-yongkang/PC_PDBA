import torch
import torch.nn as nn
import torch.nn.functional as F
from .module_helper import ModuleHelper

'''
(proj): Sequential(
        (0): Conv2d(1024, 1024, kernel_size=(1, 1), stride=(1, 1))
        (1): Sequential(
          (0): BatchNorm2d(1024, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (1): ReLU()
        )
        (2): Conv2d(1024, 256, kernel_size=(1, 1), stride=(1, 1))
'''
class ProjectionHead(nn.Module):
    def __init__(self, dim_in, proj_dim=256, proj='convmlp', bn_type='torchbn'):
        super(ProjectionHead, self).__init__()

        if proj == 'linear':
            self.proj = nn.Conv2d(dim_in, proj_dim, kernel_size=1)
        elif proj == 'convmlp':
            self.proj = nn.Sequential(
                nn.Conv2d(dim_in, dim_in, kernel_size=1),
                ModuleHelper.BNReLU(dim_in, bn_type=bn_type),
                nn.Conv2d(dim_in, proj_dim, kernel_size=1)
            )

    def forward(self, x):
        return F.normalize(self.proj(x), p=2, dim=1)