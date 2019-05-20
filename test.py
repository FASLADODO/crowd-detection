# -*- coding:utf-8 -*-

import os
import torch
import numpy as np

# from src.crowd_count import CrowdCounter
# from src.crowd_count_mod_loss import CrowdCounter
from src.crowd_count_mod_loss import CrowdCounter_cnterr_l1_out

from src.crowd_count_mod_loss import CrowdCounter_cnterr_l1_out, CrowdCounter_cnterr_LP, CrowdCounter_cnterr_LA

from src.timer import Timer
from src import utils
from src.evaluate_model import evaluate_model


from src import network
from src.data_loader import ImageDataLoader
from src import utils


torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
vis = False
save_output = True


image_folder = 'E:/processed datasets for counting'

#data_path = image_folder + '/data/original/shanghaitech/part_A_final/test_data/test_rgb/'
#gt_path = image_folder + '/data/original/shanghaitech/part_A_final/test_data/ground_truth_csv/'

data_path = 'E:/processed datasets for counting/data/formatted_trainval/mall_dataset/rgb_test/'
gt_path = 'E:/processed datasets for counting/data/formatted_trainval/mall_dataset/rgb_test_den/'

# data_path = 'E:/processed datasets for counting/data/formatted_trainval/UCSD/rgb_img/test'
# gt_path = 'E:/processed datasets for counting/data/formatted_trainval/UCSD/test_den'

current_dir = 'C:/Users/jalee/Desktop/FYP/'

# model_path
# model_path = './final_models/shtechA/done/acspnet_baseline_shtechA_249_crop_9.h5'
# model_path = current_dir + '/saved_models/schtechA/MCNN-ver1/MCNN-ver1_schtechA_43_crop_50.h5'
# model_path = 'C:/Users/jalee/Desktop/FYP/saved_models/schtechA/MCNN-ver2/MCNN-ver2_schtechA_54_crop_50.h5'
# model_path = 'C:/Users/jalee/Desktop/FYP/saved_models/schtechA/MCNN-ver3/MCNN-ver3_schtechA_35_crop_50.h5'
# model_path = 'C:/Users/jalee/Desktop/FYP/saved_models/UCSD/MCNN-ver1/MCNN-ver1_UCSD_42_crop_50.h5'
# model_path = 'C:/Users/jalee/Desktop/FYP/saved_models/mall/MCNN-final-1/MCNN-final-1_mall_6_crop_50.h5'
model_path = 'C:/Users/jalee/Desktop/FYP/saved_models/mall/MCNN-final-2/MCNN-final-2_mall_191_crop_50.h5'
# model_path = 'C:/Users/jalee/Desktop/FYP/saved_models/mall/MCNN-final-3/MCNN-final-3_mall_54_crop_50.h5'



# model_path = './final_models/shtechA/done/acspnet_ASP_shtechA_32_crop_50.h5'
# model_path = './final_models/shtechA/acspnet_ASP_shtechA_98_crop_9.h5'
# model_path = './final_models/shtechA/done/acspnet_ASP_skip_shtechA_176_crop_9.h5'
# model_path = './final_models/shtechA/done/acspnet_shtechA_168_crop_50.h5'

output_dir = current_dir + '/output/'
model_name = os.path.basename(model_path).split('.')[0]

file_results = os.path.join(output_dir, 'results_' + model_name + '.txt')
file_results_gt = os.path.join(output_dir, 'gt_results_' + model_name + '.txt')
file_results_est = os.path.join(output_dir, 'est_results_' + model_name + '.txt')

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
output_dir = os.path.join(output_dir, 'density_maps_' + model_name)

# net = CrowdCounter()
#net = CrowdCounter_cnterr_l1_out()
net = CrowdCounter_cnterr_LP()

trained_model = os.path.join(model_path)
network.load_net(trained_model, net)
net.cuda()
net.eval()
mae = 0.0
mse = 0.0

# load test data
# downsample = True
# data_loader = ImageDataLoader(data_path, gt_path, shuffle=False, gt_downsample=True, pre_load=False)
# gt_downsample = False, downsample=True
# data_loader = ImageDataLoader(data_path, gt_path, shuffle=False, gt_downsample=False, downsample=True, pre_load=False)
# gt_downsample = False, downsample=False
data_loader = ImageDataLoader(data_path, gt_path, shuffle=False, gt_downsample=False, pre_load=False)

f = open(file_results, 'w')
# f_gt = open(file_results_gt, 'w')
# f_est = open(file_results_est, 'w')

with torch.no_grad():
    for blob in data_loader:
        fname = blob['fname']
        im_data = blob['data']
        gt_data = blob['gt_density']

        im_data = im_data.cuda()
        gt_data = gt_data.cuda()

        density_map = net(im_data, gt_data)

        gt_data = gt_data.data.cpu().numpy()
        gt_count = np.sum(gt_data)
        density_map = density_map.data.cpu().numpy()
        et_count = np.sum(density_map)
        mae += abs(gt_count - et_count)
        mse += ((gt_count-et_count)*(gt_count-et_count))
        print('\nNumber %s:  \tgt_cnt: %4.1f, \tet_cnt: %4.1f, \tMAE: %4.1f' % (fname, gt_count, et_count, abs(gt_count - et_count)))

        f.write('\nNumber %s:  \tgt_cnt: %4.1f, \tet_cnt: %4.1f, \tMAE: %4.1f' % (fname, gt_count, et_count, abs(gt_count - et_count)))

        # f_gt.write('Number %s: %4.1f\n' % (fname, gt_count))
        # f_est.write('Number %s: %4.1f\n' % (fname, et_count))


        # if vis:
        #     utils.display_results(im_data, gt_data, density_map)
        # if save_output:
        #     utils.save_demo_density_map(density_map, output_dir,
        #                                 'output_' + blob['fname'].split('.')[0] + '_' + '.png')
        #     utils.save_gt(gt_data, output_dir, 'gt_' + blob['fname'].split('.')[0] + '.png')


mae = mae / data_loader.get_num_samples()
mse = np.sqrt(mse/data_loader.get_num_samples())
print('\nTOTAL MAE: %0.2f, \tTOTAL MSE: %0.2f' % (mae, mse))
f.write('\nTOTAL MAE: %0.2f, \tTOTAL MSE: %0.2f' % (mae, mse))
# print('TOTAL MAE: %0.2f' % mae)

f.close()
# f_est.close()
# f_gt.close()