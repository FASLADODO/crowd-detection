# -*- coding:utf-8 -*-

# from src.crowd_count import CrowdCounter
from src.crowd_count_mod_loss import CrowdCounter
# from src.crowd_count_mod_loss import CrowdCounter_counterr
# from src.crowd_count_mod_loss import CrowdCounter_cnterr_l1
from src.crowd_count_mod_loss import CrowdCounter_cnterr_l1_out_minibatch, CrowdCounter_cnterr_l1_out, \
    CrowdCounter_cnterr_LP, CrowdCounter_cnterr_LA
import torch.nn
from src import network
import numpy as np
import torch


def evaluate_model(trained_model, data_loader):
    #net = CrowdCounter_cnterr_l1_out()
    net = CrowdCounter_cnterr_LP()
    network.load_net(trained_model, net)

    net.cuda()
    net.eval()
    mae = 0.0
    mse = 0.0

    # no gradient calculation
    with torch.no_grad():
        for blob in data_loader:
            im_data = blob['data']
            gt_data = blob['gt_density']
            im_data = im_data.cuda()
            gt_data = gt_data.cuda()
            density_map = net(im_data, gt_data)
            density_map = density_map.data.cpu().numpy()
            gt_data = gt_data.data.cpu().numpy()
            gt_count = np.sum(gt_data)
            et_count = np.sum(density_map)
            mae += abs(gt_count-et_count)
            mse += ((gt_count-et_count)*(gt_count-et_count))
        mae = mae/data_loader.get_num_samples()
        mse = np.sqrt(mse/data_loader.get_num_samples())
        return mae,mse

def evaluate_model_minibatch(trained_model, data_loader):
    # net = CrowdCounter()
    # net = CrowdCounter_counterr()
    # net = CrowdCounter_cnterr_l1()
    net = CrowdCounter_cnterr_l1_out_minibatch()
    # net = torch.nn.DataParallel(CrowdCounter_cnterr_l1_out(), device_ids=[4, 5])

    network.load_net(trained_model, net)

    net.cuda()

    net.eval()
    mae = 0.0
    mse = 0.0
    for blob in data_loader:
        im_data = blob['data']
        gt_data = blob['gt_density']
        density_map = net(im_data, gt_data)
        density_map = density_map.data.cpu().numpy()
        gt_count = np.sum(gt_data)
        et_count = np.sum(density_map)
        mae += abs(gt_count-et_count)
        mse += ((gt_count-et_count)*(gt_count-et_count))
    mae = mae/data_loader.get_num_samples()
    mse = np.sqrt(mse/data_loader.get_num_samples())
    return mae,mse