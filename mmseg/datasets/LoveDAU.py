# ---------------------------------------------------------------
# Copyright (c) 2021-2022 ETH Zurich, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

from . import CityscapesDataset
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class LoveDAUDataset(CustomDataset):
    #CLASSES = CityscapesDataset.CLASSES
    #PALETTE = CityscapesDataset.PALETTE
    CLASSES = [ 'background','building','road','water','barren','forest','agriculture']

    PALETTE = [(255,255,255),  (0,0,255),  (0,255,255),  (0,255,0),  (255,255,0),  (255,0,0),(255,165,95)]
    
    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='_labelTrainIds.png',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)