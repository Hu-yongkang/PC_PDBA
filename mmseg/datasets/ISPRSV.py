# ISPRSPDataset
# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from . import CityscapesDataset
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class ISPRSVDataset(CustomDataset):
 
    CLASSES =  ['background','car','tree','low vegetation','building','impervious surfaces']
 				 

    PALETTE = [(255,255,255),  (0,0,255),  (0,255,255),  (0,255,0),  (255,255,0),  (255,0,0)]
   
    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix='.png',
            reduce_zero_label=reduce_zero_label,
            **kwargs)