import torch.nn as nn
from src import network
import numpy as np

from src.model import MCNN
from src.model import acipnet

class CrowdCounter(nn.Module):
    def __init__(self):
        super(CrowdCounter, self).__init__()        
        self.DME = MCNN()
        # self.DME = acipnet()
        self.loss_mse = nn.MSELoss(size_average=False)

    @property
    def loss(self):
        return self.loss_all
    
    def forward(self,  im_data, gt_data=None, is_training=False):
        # im_data = im_data.cuda()
        density_map = self.DME(im_data)

        if is_training:
            # gt_data = gt_data.cuda()
            self.loss_all = self.build_loss(density_map, gt_data)
            
        return density_map
    
    def build_loss(self, density_map, gt_data):
        # loss = self.loss_fn(density_map, gt_data)
        loss_mse = 0.5 * self.loss_mse(density_map, gt_data)
        return loss_mse


class CrowdCounter_counterr(nn.Module):
    def __init__(self):
        super(CrowdCounter_counterr, self).__init__()
        self.DME = MCNN()

        self.alpha = 0.0001
        self.loss_mse = nn.MSELoss(size_average=False)

    @property
    def loss(self):
        return self.loss_all

    def forward(self, im_data, gt_data=None):
        im_data = network.np_to_variable(im_data, is_cuda=True, is_training=self.training)
        density_map = self.DME(im_data)

        if self.training:
            gt_data = network.np_to_variable(gt_data, is_cuda=True, is_training=self.training)
            self.loss_all = self.build_loss(density_map, gt_data)
            # self.loss_mse = self.build_loss(density_map, gt_data)

        return density_map

    def build_loss(self, density_map, gt_data):
        # loss = self.loss_fn(density_map, gt_data)
        loss_mse = 0.5 * self.loss_mse(density_map, gt_data)
        # density_map = density_map.data.cpu().numpy()
        # gt_data = gt_data.data.cpu().numpy()
        gt_count = gt_data.sum()
        et_count = density_map.sum()
        if gt_count.data.cpu().numpy() > et_count.data.cpu().numpy():
            loss_count = self.alpha * (gt_count-et_count)
        else:
            loss_count = self.alpha * (et_count-gt_count)
        return loss_mse + loss_count


class CrowdCounter_cnterr_l1(nn.Module):
    def __init__(self):
        super(CrowdCounter_cnterr_l1, self).__init__()
        self.DME = MCNN()

        self.alpha_1 = 0.0001
        self.alpha_2 = 0.0001

        # self.beta_1 = 0.1
        # self.beta_2 = 0.01

        self.loss_mse = nn.MSELoss(size_average=False)
        self.loss_L1 = nn.L1Loss(size_average=False)

    @property
    def loss(self):
        return self.loss_all

    def forward(self, im_data, gt_data=None):
        im_data = network.np_to_variable(im_data, is_cuda=True, is_training=self.training)
        density_map = self.DME(im_data)

        if self.training:
            gt_data = network.np_to_variable(gt_data, is_cuda=True, is_training=self.training)
            self.loss_all = self.build_loss(density_map, gt_data)
            # self.loss_mse = self.build_loss(density_map, gt_data)

        return density_map[2]

    def build_loss(self, density_map, gt_data):
        # loss = self.loss_fn(density_map, gt_data)
        loss_mse_x = 0.5 * self.loss_mse(density_map[0], gt_data)
        loss_L1_x = self.alpha_2 * self.loss_L1(density_map[0], gt_data)

        loss_mse_rec1 = 0.5 * self.loss_mse(density_map[1], gt_data)
        loss_L1_rec1 = self.alpha_2 * self.loss_L1(density_map[1], gt_data)

        loss_mse_rec2 = 0.5 * self.loss_mse(density_map[2], gt_data)
        loss_L1_rec2 = self.alpha_2 * self.loss_L1(density_map[2], gt_data)

        # density_map = density_map.data.cpu().numpy()
        # gt_data = gt_data.data.cpu().numpy()
        gt_count = gt_data.sum()

        et_count_x = density_map[0].sum()
        et_count_rec1 = density_map[1].sum()
        et_count_rec2 = density_map[2].sum()

        if gt_count.data.cpu().numpy() > et_count_x.data.cpu().numpy():
            loss_count_x = self.alpha_1 * (gt_count-et_count_x)
        else:
            loss_count_x = self.alpha_1 * (et_count_x-gt_count)

        if gt_count.data.cpu().numpy() > et_count_rec1.data.cpu().numpy():
            loss_count_rec1 = self.alpha_1 * (gt_count-et_count_rec1)
        else:
            loss_count_rec1 = self.alpha_1 * (et_count_rec1-gt_count)

        if gt_count.data.cpu().numpy() > et_count_rec2.data.cpu().numpy():
            loss_count_rec2 = self.alpha_1 * (gt_count-et_count_rec2)
        else:
            loss_count_rec2 = self.alpha_1 * (et_count_rec2-gt_count)


        # return self.beta_2 * (loss_mse_x + loss_count_x + loss_L1_x) + \
        #        self.beta_1 * (loss_mse_rec1 + loss_count_rec1 + loss_L1_rec1) + \
        #        (loss_mse_rec2 + loss_count_rec2 + loss_L1_rec2)

        return (loss_mse_x + loss_count_x + loss_L1_x) + \
                (loss_mse_rec1 + loss_count_rec1 + loss_L1_rec1) + \
                (loss_mse_rec2 + loss_count_rec2 + loss_L1_rec2)

        # return loss_mse_x + (loss_mse_rec1 + loss_count_rec1) + (loss_mse_rec2 + loss_count_rec2 + loss_L1_rec2)

class CrowdCounter_cnterr_l1_out(nn.Module):
    def __init__(self):
        super(CrowdCounter_cnterr_l1_out, self).__init__()
        self.DME = MCNN()
        # self.DME = acipnet()
        self.alpha_1 = 0.001
        # self.alpha_2 = 0.001
        self.alpha_2 = 0.0001

# ******************************************************
        # single img input w/o minibatch
        self.loss_mse = nn.MSELoss(size_average=False)
        self.loss_L1 = nn.L1Loss(size_average=False)

        # minibatch
        # self.loss_mse = nn.MSELoss()
        # self.loss_L1 = nn.L1Loss()
# ******************************************************

    @property
    def loss(self):
        return self.loss_all

    def forward(self, im_data, gt_data=None, is_training=False):
        # im_data = im_data.cuda()
        density_map = self.DME(im_data)

        if is_training:
            # gt_data = gt_data.cuda()
            self.loss_all = self.build_loss(density_map, gt_data)

        return density_map

    def build_loss(self, density_map, gt_data):
        loss_mse_rec2 = 0.5 * self.loss_mse(density_map, gt_data)
        # loss_mse_rec2 = self.loss_mse(density_map, gt_data)
        loss_L1_rec2 = self.alpha_2 * self.loss_L1(density_map, gt_data)

        density_map = density_map.data.cpu().numpy()
        gt_data = gt_data.data.cpu().numpy()
        gt_count = np.sum(gt_data)
        et_count_rec2 = np.sum(density_map)

        # if gt_count.data.cpu().numpy() > et_count_rec2.data.cpu().numpy():
        #     loss_count_rec2 = self.alpha_1 * (gt_count-et_count_rec2)
        # else:
        #     loss_count_rec2 = self.alpha_1 * (et_count_rec2-gt_count)
        loss_count_rec2 = self.alpha_1 * abs(gt_count-et_count_rec2)

        return (loss_mse_rec2 + loss_count_rec2 + loss_L1_rec2)


class CrowdCounter_cnterr_LA(nn.Module):
    def __init__(self):
        super(CrowdCounter_cnterr_LA, self).__init__()
        self.DME = MCNN()
        # self.DME = acipnet()

        self.alpha_1 = 0.001
        self.alpha_2 = 0.0001

        # single img input w/o minibatch
        self.loss_mse = nn.MSELoss(size_average=False)
        # self.loss_L1 = nn.L1Loss(size_average=False)

    @property
    def loss(self):
        return self.loss_all

    def forward(self, im_data, gt_data=None, is_training=False):
        # im_data = im_data.cuda()
        density_map = self.DME(im_data)

        if is_training:
            # gt_data = gt_data.cuda()
            self.loss_all = self.build_loss(density_map, gt_data)

        return density_map

    def build_loss(self, density_map, gt_data):
        loss_mse_rec2 = 0.5 * self.loss_mse(density_map, gt_data)
        density_map = density_map.data.cpu().numpy()
        gt_data = gt_data.data.cpu().numpy()
        gt_count = np.sum(gt_data)
        et_count_rec2 = np.sum(density_map)

        loss_count_rec2 = self.alpha_1 * abs(gt_count-et_count_rec2)

        return (loss_mse_rec2 + loss_count_rec2)


class CrowdCounter_cnterr_LP(nn.Module):
    def __init__(self):
        super(CrowdCounter_cnterr_LP, self).__init__()
        self.DME = MCNN()
        # self.DME = acipnet()
        self.alpha_1 = 0.001
        self.alpha_2 = 0.0001

        # single img input w/o minibatch
        self.loss_mse = nn.MSELoss(size_average=False)
        self.loss_L1 = nn.L1Loss(size_average=False)

    @property
    def loss(self):
        return self.loss_all

    def forward(self, im_data, gt_data=None, is_training=False):
        # im_data = im_data.cuda()
        density_map = self.DME(im_data)

        if is_training:
            # gt_data = gt_data.cuda()

            self.loss_all = self.build_loss(density_map, gt_data)

        return density_map

    def build_loss(self, density_map, gt_data):
        loss_mse_rec2 = 0.5 * self.loss_mse(density_map, gt_data)
        # loss_mse_rec2 = self.loss_mse(density_map, gt_data)
        loss_L1_rec2 = self.alpha_2 * self.loss_L1(density_map, gt_data)

        return (loss_mse_rec2 + loss_L1_rec2)

class CrowdCounter_cnterr_l1_out_minibatch(nn.Module):
    def __init__(self):
        super(CrowdCounter_cnterr_l1_out_minibatch, self).__init__()
        # self.DME = ms_skip_global()
        # self.DME = ms_skip_global_rec_deconv()
        # self.DME = ms_skip_global_rec_deconv_global()
        # self.DME = ms_pool_deconv_skip()
        # self.DME = ms_hourglass_skip_triple()

        self.alpha_1 = 0.001
        self.alpha_2 = 0.0001

        # ******************************************************
        # single img input w/o minibatch
        #         self.loss_mse = nn.MSELoss(size_average=False)
        #         self.loss_L1 = nn.L1Loss(size_average=False)

        # minibatch
        self.loss_mse = nn.MSELoss()
        self.loss_L1 = nn.L1Loss()
        # ******************************************************

    @property
    def loss(self):
        return self.loss_all

    def forward(self, im_data, gt_data=None, is_training=False):
        # --------------------------------------------------------------------------------------------------
        # single img input w/o minibatch
        # im_data = network.np_to_variable(im_data, is_cuda=True, is_training=is_training)

        # minibatch
        im_data = network.np_to_variable_minibatch(im_data, is_cuda=True, is_density=False, is_training=is_training)
        # --------------------------------------------------------------------------------------------------
        density_map = self.DME(im_data)

        if self.training:
            # --------------------------------------------------------------------------------------------------
            # single img input w/o minibatch
            # gt_data = network.np_to_variable(gt_data, is_cuda=True, is_training=is_training)

            # minibatch
            gt_data = network.np_to_variable_minibatch(gt_data, is_cuda=True, is_density=True, is_training=is_training)
            # --------------------------------------------------------------------------------------------------

            self.loss_all = self.build_loss(density_map, gt_data)
            # self.loss_mse = self.build_loss(density_map, gt_data)

        return density_map

    def build_loss(self, density_map, gt_data):

        loss_mse_rec2 = 0.5 * self.loss_mse(density_map, gt_data)
        loss_L1_rec2 = self.alpha_2 * self.loss_L1(density_map, gt_data)

        # density_map = density_map.data.cpu().numpy()
        # gt_data = gt_data.data.cpu().numpy()
        gt_count = gt_data.sum()
        et_count_rec2 = density_map.sum()

        if gt_count.data.cpu().numpy() > et_count_rec2.data.cpu().numpy():
            loss_count_rec2 = self.alpha_1 * (gt_count - et_count_rec2)
        else:
            loss_count_rec2 = self.alpha_1 * (et_count_rec2 - gt_count)

        return (loss_mse_rec2 + loss_count_rec2 + loss_L1_rec2)
