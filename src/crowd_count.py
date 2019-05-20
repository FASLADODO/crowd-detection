# -*- coding:utf-8 -*-

import torch.nn as nn
from src import network
from src.model import MCNN
# from src.model_multi_ms import MMSCNN
# from src.model_mscnn import MSCNN

class CrowdCounter(nn.Module):
    def __init__(self):
        super(CrowdCounter, self).__init__()

        self.DME = MCNN()
        # self.DME = MMSCNN()
        # self.DME = MSCNN()

        self.loss_fn = nn.MSELoss()
        
    @property
    def loss(self):
        return self.loss_mse
    
    def forward(self, im_data, gt_data=None):
        im_data = network.np_to_variable(im_data, is_cuda=True, is_training=self.training)
        density_map = self.DME(im_data)
        
        if self.training:                        
            gt_data = network.np_to_variable(gt_data, is_cuda=True, is_training=self.training)            
            self.loss_mse = self.build_loss(density_map, gt_data)
            
        return density_map
    
    def build_loss(self, density_map, gt_data):
        loss = self.loss_fn(density_map, gt_data)
        # loss = 0.5 * self.loss_fn(density_map, gt_data)
        return loss
