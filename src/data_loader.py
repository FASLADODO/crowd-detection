# -*- coding:utf-8 -*-
import torch
import numpy as np
import cv2
import os
import random
import pandas as pd
from torch.autograd import Variable

class ImageDataLoader():
    def __init__(self, data_path, gt_path, shuffle=False, gt_downsample=False, downsample=False, pre_load=True):
        # pre_load: if true, all training and validation images are loaded into CPU RAM for faster processing.
        #          This avoids frequent file reads. Use this only for small datasets.
        self.data_path = data_path
        self.gt_path = gt_path
        self.downsample = downsample
        self.gt_downsample = gt_downsample
        self.pre_load = pre_load
        self.data_files = [filename for filename in os.listdir(data_path) \
                           if os.path.isfile(os.path.join(data_path,filename))]
        self.data_files.sort()
        self.shuffle = shuffle
        if shuffle:
            random.seed(2468)
        self.num_samples = len(self.data_files)
        # print(self.num_samples)
        self.blob_list = {}        
        self.id_list = list(range(0,self.num_samples))
        if self.pre_load:
            print('Pre-loading the data. This may take a while...')
            idx = 0
            for fname in self.data_files:
                
                # img = cv2.imread(os.path.join(self.data_path,fname),0)
                img = cv2.imread(os.path.join(self.data_path, fname))
                img = img.astype(np.float32, copy=False)


                # ---------------------------------------------------
                if self.gt_downsample:
                    ht = img.shape[0]
                    wd = img.shape[1]
                    # ht_1 = int(ht/2)*2
                    # wd_1 = int(wd/2)*2
                    # ---------
                    ht_1 = int(ht/4)*4
                    wd_1 = int(wd/4)*4
                    # ---------
                    # ht_1 = int(ht/8)*8
                    # wd_1 = int(wd/8)*8
                    img = cv2.resize(img,(wd_1,ht_1))

                # img = img.reshape((1,1,img.shape[0],img.shape[1]))
                # ---------------------------------------------------

                den = pd.read_csv(os.path.join(self.gt_path,os.path.splitext(fname)[0] + '.csv'), sep=',',header=None).as_matrix()                        
                den = den.astype(np.float32, copy=False)

                # ---------------------------------------------------
                if self.gt_downsample:
                    # no deconv layer --------------
                    # wd_1 = int(wd_1/2)
                    # ht_1 = int(ht_1/2)
                    # ---------
                    wd_1 = int(wd_1/4) * 4
                    ht_1 = int(ht_1/4) * 4
                    # ---------
                    # wd_1 = int(wd_1/8)
                    # ht_1 = int(ht_1/8)
                    # ------------------------------
                    den = cv2.resize(den,(wd_1,ht_1))
                    den = den * ((wd*ht)/(wd_1*ht_1))
                # else:
                #     den = cv2.resize(den,(wd_1,ht_1))
                #     den = den * ((wd*ht)/(wd_1*ht_1))

                # den = den.reshape((1,1,den.shape[0],den.shape[1]))
                # ---------------------------------------------------

                # ---------------------------------------------------
                if self.downsample:
                    ht = img.shape[0]
                    wd = img.shape[1]

                    ht_1 = int(ht_1/2)
                    wd_1 = int(wd_1/2)

                    # ht_1 = int(ht_1/4)
                    # wd_1 = int(wd_1/4)

                    img = cv2.resize(img,(wd_1,ht_1))

                # img = img.reshape((1,1,img.shape[0],img.shape[1]))
                # ---------------------------------------------------

                # den = pd.read_csv(os.path.join(self.gt_path,os.path.splitext(fname)[0] + '.csv'), sep=',',header=None).as_matrix()
                # den = den.astype(np.float32, copy=False)

                if self.downsample:
                    den = cv2.resize(den,(wd_1,ht_1))
                    den = den * ((wd*ht)/(wd_1*ht_1))

                # change rgb channel
                img = np.transpose(img, (2, 0, 1))
                img = img[np.newaxis, :]
                den = den[np.newaxis, np.newaxis, :]
                img = torch.from_numpy(img)

                # img = Variable(img)
                # img = img.cuda()
                den = torch.from_numpy(den)

                # den = Variable(den)
                # den = den.cuda()

                blob = {}
                blob['data'] = img
                blob['gt_density'] = den
                blob['fname'] = fname
                self.blob_list[idx] = blob
                idx = idx+1
                if idx % 5000 == 0:
                    print('Loaded ', idx, '/', self.num_samples, 'files')
                    # print(np.shape(den))
                    # print(np.shape(img))
               
            print('Completed Loading ', idx, 'files')
        
        
    def __iter__(self):
        if self.shuffle:
            if self.pre_load:
                random.shuffle(self.id_list)
            else:
                random.shuffle(self.data_files)
        files = self.data_files
        id_list = self.id_list

        for idx in id_list:
            if self.pre_load:
                blob = self.blob_list[idx]
                blob['idx'] = idx
            else:
                fname = files[idx]
                img = cv2.imread(os.path.join(self.data_path,fname))
                img = img.astype(np.float32, copy=False)

                # -----------------------------------------------------
                if self.gt_downsample:
                    ht = img.shape[0]
                    wd = img.shape[1]
                    # ht_1 = int(ht / 2) * 2
                    # wd_1 = int(wd / 2) * 2
                    # ---------
                    ht_1 = int(ht/4)*4
                    wd_1 = int(wd/4)*4
                    # ---------
                    # ht_1 = int(ht/8)*8
                    # wd_1 = int(wd/8)*8
                    img = cv2.resize(img,(wd_1,ht_1))
                # ht = img.shape[0]
                # wd = img.shape[1]
                # ht_1 = (ht/4)*4
                # wd_1 = (wd/4)*4
                # img = cv2.resize(img,(wd_1,ht_1))

                # img = img.reshape((1,1,img.shape[0],img.shape[1]))
                # -----------------------------------------------------

                den = pd.read_csv(os.path.join(self.gt_path,os.path.splitext(fname)[0] + '.csv'), sep=',',header=None).as_matrix()
                den = den.astype(np.float32, copy=False)

                # -----------------------------------------------------
                if self.gt_downsample:
                    den = cv2.resize(den,(wd_1,ht_1))
                    den = den * ((wd*ht)/(wd_1*ht_1))
                # else:
                #     den = cv2.resize(den,(wd_1,ht_1))
                #     den = den * ((wd*ht)/(wd_1*ht_1))

                # den = den.reshape((1,1,den.shape[0],den.shape[1]))
                # -----------------------------------------------------

                # ---------------------------------------------------
                if self.downsample:
                    ht = img.shape[0]
                    wd = img.shape[1]
                    ht_1 = int(ht_1/2)
                    wd_1 = int(wd_1/2)

                    # wd_1 = int(wd_1/4)
                    # ht_1 = int(ht_1/4)

                    img = cv2.resize(img,(wd_1,ht_1))

                # img = img.reshape((1,1,img.shape[0],img.shape[1]))
                # ---------------------------------------------------

                # den = pd.read_csv(os.path.join(self.gt_path,os.path.splitext(fname)[0] + '.csv'), sep=',',header=None).as_matrix()
                # den = den.astype(np.float32, copy=False)

                if self.downsample:
                    den = cv2.resize(den,(wd_1,ht_1))
                    den = den * ((wd*ht)/(wd_1*ht_1))

                # change rgb channel
                img = np.transpose(img, (2, 0, 1))
                img = img[np.newaxis, :]
                den = den[np.newaxis, np.newaxis, :]
                img = torch.from_numpy(img)

                # img = Variable(img)
                # img = img.cuda()
                den = torch.from_numpy(den)

                # den = Variable(den)
                # den = den.cuda()

                blob = {}
                blob['data'] = img
                blob['gt_density'] = den
                blob['fname'] = fname

            yield blob

    def get_num_samples(self):
        return self.num_samples


class ImageDataLoader_minibatch():
    def __init__(self, data_path, gt_path, batch_size, shuffle=False, gt_downsample=False, pre_load=True):
        # pre_load: if true, all training and validation images are loaded into CPU RAM for faster processing.
        #          This avoids frequent file reads. Use this only for small datasets.
        self.data_path = data_path
        self.gt_path = gt_path
        self.gt_downsample = gt_downsample
        self.pre_load = pre_load
        self.data_files = [filename for filename in os.listdir(data_path) \
                           if os.path.isfile(os.path.join(data_path, filename))]
        self.data_files.sort()
        self.shuffle = shuffle
        if shuffle:
            random.seed(2468)
        self.num_samples = len(self.data_files)
        # print(self.num_samples)
        self.blob_list = {}
        self.id_list = list(range(0, self.num_samples))
        self.batch_size = batch_size
        if self.pre_load:
            print('Pre-loading the data. This may take a while...')
            idx = 0
            for fname in self.data_files:

                # img = cv2.imread(os.path.join(self.data_path,fname),0)
                img = cv2.imread(os.path.join(self.data_path, fname))
                img = img.astype(np.float32, copy=False)

                # ---------------------------------------------------
                if self.gt_downsample:
                    ht = img.shape[0]
                    wd = img.shape[1]
                    ht_1 = int(ht / 4) * 4
                    wd_1 = int(wd / 4) * 4
                    img = cv2.resize(img, (wd_1, ht_1))

                # img = img.reshape((1,1,img.shape[0],img.shape[1]))
                # ---------------------------------------------------

                den = pd.read_csv(os.path.join(self.gt_path, os.path.splitext(fname)[0] + '.csv'), sep=',',
                                  header=None).as_matrix()
                den = den.astype(np.float32, copy=False)

                # ---------------------------------------------------
                if self.gt_downsample:
                    # no deconv layers
                    # wd_1 = int(wd_1 / 4)
                    # ht_1 = int(ht_1 / 4)

                    den = cv2.resize(den, (wd_1, ht_1))
                    den = den * ((wd * ht) / (wd_1 * ht_1))
                # else:
                #     den = cv2.resize(den,(wd_1,ht_1))
                #     den = den * ((wd*ht)/(wd_1*ht_1))

                # den = den.reshape((1,1,den.shape[0],den.shape[1]))
                # ---------------------------------------------------
                # img = np.transpose(img, (2, 0, 1))

                blob = {}
                blob['data'] = img
                blob['gt_density'] = den
                blob['fname'] = fname
                self.blob_list[idx] = blob
                idx = idx + 1
                if idx % 5000 == 0:
                    print('Loaded ', idx, '/', self.num_samples, 'files')
                    # print(np.shape(den))
                    # print(np.shape(img))

            print('Completed Loading ', idx, 'files')

    def __iter__(self):
        if self.shuffle:
            if self.pre_load:
                random.shuffle(self.id_list)
            else:
                random.shuffle(self.data_files)

        files = self.data_files
        id_list = self.id_list
        idx_list = []
        step = 0
        for n in range(int(len(id_list) / self.batch_size) + 1):
            if self.pre_load:
                new_blob = {}
                new_blob['data'] = []
                new_blob['gt_density'] = []
                step = step + 1
                # Mini-batch test
                if n * self.batch_size + self.batch_size > self.num_samples:
                    for i in range(n * self.batch_size, self.num_samples):
                        batch_idx = id_list[i]
                        new_blob['data'].append(self.blob_list[batch_idx]['data'])
                        new_blob['gt_density'].append(self.blob_list[batch_idx]['gt_density'])
                else:
                    for i in range(n * self.batch_size, n * self.batch_size + self.batch_size):
                        batch_idx = id_list[i]
                        new_blob['data'].append(self.blob_list[batch_idx]['data'])
                        new_blob['gt_density'].append(self.blob_list[batch_idx]['gt_density'])
                yield new_blob

    def get_num_samples(self):
        return self.num_samples


class ImageDataLoader_without_den():
    def __init__(self, data_path, gt_path=None, shuffle=False, gt_downsample=False, pre_load=True):
        # pre_load: if true, all training and validation images are loaded into CPU RAM for faster processing.
        #          This avoids frequent file reads. Use this only for small datasets.
        self.data_path = data_path
        self.gt_path = gt_path
        self.gt_downsample = gt_downsample
        self.pre_load = pre_load
        self.data_files = [filename for filename in os.listdir(data_path) \
                           if os.path.isfile(os.path.join(data_path, filename))]
        self.data_files.sort()
        self.shuffle = shuffle
        if shuffle:
            random.seed(2468)
        self.num_samples = len(self.data_files)
        # print(self.num_samples)
        self.blob_list = {}
        self.id_list = list(range(0, self.num_samples))
        if self.pre_load:
            print('Pre-loading the data. This may take a while...')
            idx = 0
            for fname in self.data_files:

                # img = cv2.imread(os.path.join(self.data_path,fname),0)
                img = cv2.imread(os.path.join(self.data_path, fname))
                img = img.astype(np.float32, copy=False)

                # ---------------------------------------------------
                if self.gt_downsample:
                    ht = img.shape[0]
                    wd = img.shape[1]
                    ht_1 = int(ht / 4) * 4
                    wd_1 = int(wd / 4) * 4
                    # ht_1 = int(ht / 32) * 8
                    # wd_1 = int(wd / 32) * 8
                    img = cv2.resize(img, (wd_1, ht_1))

                # img = img.reshape((1,1,img.shape[0],img.shape[1]))
                # ---------------------------------------------------

                # change rgb channel
                img = np.transpose(img, (2, 0, 1))
                img = img[np.newaxis, :]
                img = torch.from_numpy(img)
                img = Variable(img)
                img = img.cuda()

                blob = {}
                blob['data'] = img

                if (self.gt_path is not None):
                    den = pd.read_csv(os.path.join(self.gt_path, os.path.splitext(fname)[0] + '.csv'), sep=',',
                                  header=None).as_matrix()
                    den = den.astype(np.float32, copy=False)

                # ---------------------------------------------------
                    if self.gt_downsample:
                        # wd_1 = int(wd_1/4)
                        # ht_1 = int(ht_1/4)
                        den = cv2.resize(den, (wd_1, ht_1))
                        den = den * ((wd * ht) / (wd_1 * ht_1))
                    # else:
                    #     den = cv2.resize(den,(wd_1,ht_1))
                    #     den = den * ((wd*ht)/(wd_1*ht_1))

                    # den = den.reshape((1,1,den.shape[0],den.shape[1]))
                # ---------------------------------------------------
                    blob['gt_density'] = den


                blob['fname'] = fname
                self.blob_list[idx] = blob
                idx = idx + 1
                if idx % 5000 == 0:
                    print('Loaded ', idx, '/', self.num_samples, 'files')
                    # print(np.shape(den))
                    # print(np.shape(img))

            print('Completed Loading ', idx, 'files')

    def __iter__(self):
        if self.shuffle:
            if self.pre_load:
                random.shuffle(self.id_list)
            else:
                random.shuffle(self.data_files)
        files = self.data_files
        id_list = self.id_list

        for idx in id_list:
            if self.pre_load:
                blob = self.blob_list[idx]
                blob['idx'] = idx
            else:
                fname = files[idx]
                img = cv2.imread(os.path.join(self.data_path, fname))
                img = img.astype(np.float32, copy=False)

                # -----------------------------------------------------
                if self.gt_downsample:
                    ht = img.shape[0]
                    wd = img.shape[1]
                    ht_1 = int(ht / 4) * 4
                    wd_1 = int(wd / 4) * 4
                    # ht_1 = int(ht / 32) * 8
                    # wd_1 = int(wd / 32) * 8
                    img = cv2.resize(img, (wd_1, ht_1))
                # ht = img.shape[0]
                # wd = img.shape[1]
                # ht_1 = (ht/4)*4
                # wd_1 = (wd/4)*4
                # img = cv2.resize(img,(wd_1,ht_1))

                # img = img.reshape((1,1,img.shape[0],img.shape[1]))
                # -----------------------------------------------------

                # change rgb channel
                img = np.transpose(img, (2, 0, 1))
                img = img[np.newaxis, :]
                img = torch.from_numpy(img)
                img = Variable(img)
                img = img.cuda()

                blob = {}
                blob['data'] = img

                if (self.gt_path is not None):
                    den = pd.read_csv(os.path.join(self.gt_path, os.path.splitext(fname)[0] + '.csv'), sep=',',
                                  header=None).as_matrix()
                    den = den.astype(np.float32, copy=False)

                # ---------------------------------------------------
                    if self.gt_downsample:
                        # wd_1 = int(wd_1/4)
                        # ht_1 = int(ht_1/4)
                        den = cv2.resize(den, (wd_1, ht_1))
                        den = den * ((wd * ht) / (wd_1 * ht_1))
                    # else:
                    #     den = cv2.resize(den,(wd_1,ht_1))
                    #     den = den * ((wd*ht)/(wd_1*ht_1))

                    # den = den.reshape((1,1,den.shape[0],den.shape[1]))
                # ---------------------------------------------------
                    blob['gt_density'] = den

                blob['fname'] = fname

            yield blob

    def get_num_samples(self):
        return self.num_samples

class SingleImageDataLoader():
    def __init__(self, data_path, gt_path, shuffle=False, gt_downsample=False, pre_load=True):
        # pre_load: if true, all training and validation images are loaded into CPU RAM for faster processing.
        #          This avoids frequent file reads. Use this only for small datasets.
        self.data_path = data_path
        self.gt_path = gt_path
        self.gt_downsample = gt_downsample
        self.pre_load = pre_load
        self.num_samples =1
        img = cv2.imread(self.data_path)
        img = img.astype(np.float32, copy=False)

        # ---------------------------------------------------
        if self.gt_downsample:
            ht = img.shape[0]
            wd = img.shape[1]
            ht_1 = int(ht / 4) * 4
            wd_1 = int(wd / 4) * 4
            img = cv2.resize(img, (wd_1, ht_1))

        den = pd.read_csv(self.gt_path, sep=',', header=None).as_matrix()
        den = den.astype(np.float32, copy=False)

        # ---------------------------------------------------
        if self.gt_downsample:
            # wd_1 = int(wd_1 / 4)
            # ht_1 = int(ht_1 / 4)

            den = cv2.resize(den, (wd_1, ht_1))
            den = den * ((wd * ht) / (wd_1 * ht_1))
        self.blob = {}
        self.blob['data'] = img
        self.blob['gt_density'] = den
        self.blob['fname'] = self.data_path.split('/')[-1]

        # print('Completed Loading image files')

    def __iter__(self):
        yield self.blob