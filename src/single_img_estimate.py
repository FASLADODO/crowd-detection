# -*- coding:utf-8 -*-

import os
import torch
import numpy as np
import cv2
from src.crowd_count_mod_loss import CrowdCounter
from src.crowd_count_mod_loss import CrowdCounter_cnterr_l1_out
from src import network
from src.data_loader import ImageDataLoader,SingleImageDataLoader
from src import utils
from src.timer import Timer

t=Timer()
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
vis = False
save_output = True

# data_path = './data/original/shanghaitech/part_A_final/test_data/images/'
# gt_path = './data/original/shanghaitech/part_A_final/test_data/ground_truth_csv/'

def single_img_estimate(input_img):
    self_data_path = input_img
    # print('get data path!')
    gt_path = './data/mall_demo/rgb_den/' + self_data_path.split('/')[-1].replace('.jpg','.csv')
    # print('get gt path!')
    # branch pre-train
    model_path = './final_models/ms_pool_deconv_skip_mall_60_ms.h5'

    self_output_dir = './data/mall_demo/demo_output/'
    self_gt_dir='./data/mall_demo/demo_gt/'

    model_name = os.path.basename(model_path).split('.')[0]
    file_results = os.path.join(self_output_dir, 'results_' + model_name + '.txt')
    if not os.path.exists(self_output_dir):
        os.mkdir(self_output_dir)
    self_output_dir = os.path.join(self_output_dir, 'den_' + model_name)
    if not os.path.exists(self_output_dir):
        os.mkdir(self_output_dir)

    if not os.path.exists(self_gt_dir):
        os.mkdir(self_gt_dir)
    gt_dir = os.path.join(self_gt_dir, 'gt_' + model_name)
    if not os.path.exists(self_gt_dir):
        os.mkdir(self_gt_dir)

    # print('mkdir done!')
    net = CrowdCounter_cnterr_l1_out()

    trained_model = os.path.join(model_path)
    network.load_net(trained_model, net)
    net.cuda(3)
    net.eval()
    mae = 0.0
    mse = 0.0

    # load test data
    # downsample = True
    # data_loader = ImageDataLoader(data_path, gt_path, shuffle=False, gt_downsample=True, pre_load=False)

    # downsample = False
    data_loader = SingleImageDataLoader(self_data_path, gt_path, shuffle=False, gt_downsample=False, pre_load=False)

    for blob in data_loader:
        im_data = blob['data']
        gt_data = blob['gt_density']
        t.tic()
        density_map = net(im_data, gt_data,is_training = False)
        density_map = density_map.data.cpu().numpy()
        duration=t.toc()
        print ("time duration:"+str(duration))
        gt_count = np.sum(gt_data)
        et_count = np.sum(density_map)
        mae += abs(gt_count - et_count)
        mse += ((gt_count - et_count) * (gt_count - et_count))
        if vis:
            utils.display_results(im_data, gt_data, density_map)
        if save_output:
            utils.save_demo_density_map(density_map, self_output_dir, 'output_' + blob['fname'].split('.')[0] + '.png')

            gt_data = 255 * gt_data / np.max(gt_data)
            gt_data= gt_data.astype(np.uint8)
            gt_data = cv2.applyColorMap(gt_data,cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(self_gt_dir,'gt_'+blob['fname'].split('.')[0]+'.png'),gt_data)

        print('\nMAE: %0.2f, MSE: %0.2f' % (mae, mse))
        # f = open(file_results, 'w')
        # f.write('MAE: %0.2f, MSE: %0.2f' % (mae, mse))
        # f.close()
        # et_path = self.output_dir + 'output_' + blob['fname'].split('.')[0] + '.png'
        # gt_path = self.gt_dir+'gt_'+blob['fname'].split('.')[0]+'.png'

        return (self_output_dir + '/output_' + blob['fname'].split('.')[0] + '.png',
            self_gt_dir+'/gt_'+blob['fname'].split('.')[0]+'.png',mae,mse,gt_count,et_count)
