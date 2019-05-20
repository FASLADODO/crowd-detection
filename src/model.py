# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
from src.network import Conv2d, Dilated_Conv2D, Deconv2D
from src.network import global_pool
import torch.nn.functional as F

class MCNN(nn.Module):
    '''
    Multi-column Structure
    '''

    def __init__(self, bn=False):
        super(MCNN, self).__init__()
        self.branch1 = nn.Sequential(Conv2d( 3, 16, 9, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(16, 32, 7, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(32, 16, 7, same_padding=True, bn=bn),
                                     Conv2d(16,  8, 7, same_padding=True, bn=bn))

        self.branch2 = nn.Sequential(Conv2d( 3, 20, 7, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(20, 40, 5, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(40, 20, 5, same_padding=True, bn=bn),
                                     Conv2d(20, 10, 5, same_padding=True, bn=bn))

        self.branch3 = nn.Sequential(Conv2d( 3, 24, 5, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(24, 48, 3, same_padding=True, bn=bn),
                                     nn.MaxPool2d(2),
                                     Conv2d(48, 24, 3, same_padding=True, bn=bn),
                                     Conv2d(24, 12, 3, same_padding=True, bn=bn))

        self.fuse = nn.Sequential(Conv2d( 30, 1, 1, same_padding=True, bn=bn))


    def forward(self, im_data):
        x1 = self.branch1(im_data)
        x2 = self.branch2(im_data)
        x3 = self.branch3(im_data)

        x = torch.cat((x1, x2, x3), 1)                      # 1+2+3
        x = self.fuse(x)
        x = torch.nn.functional.upsample_bilinear(x, scale_factor=4)

        #print('x.size = ',x.size(),'\n')

        return x

class acipnet(nn.Module):
    def __init__(self, bn=False):
        super(acipnet, self).__init__()
        self.multi_scale_1 = nn.Sequential(Dilated_Conv2D(3, 64, 3, dilation=1, bn=bn),          # receptive field 3*3
                                           # nn.MaxPool2d(2),
                                           Dilated_Conv2D(64, 64, 3, dilation=1, bn=bn),
                                           Dilated_Conv2D(64, 64, 3, dilation=1, bn=bn))               # 5*5

        self.multi_scale_2 = nn.Sequential(Dilated_Conv2D(64, 64, 3, dilation=2, bn=bn),               # 9*9
                                           Dilated_Conv2D(64, 64, 3, dilation=2, bn=bn),
                                            Dilated_Conv2D(64, 64, 3, dilation=2, bn=bn))               # 17*17

        self.multi_scale_3 = nn.Sequential(Dilated_Conv2D(64, 64, 3, dilation=4, bn=bn),              # 33*33
                                           Dilated_Conv2D(64, 64, 3, dilation=4, bn=bn),              # 65*65
                                           Dilated_Conv2D(64, 64, 3, dilation=1, bn=bn))               # 67*67

        self.parallel_1 = nn.Sequential(Conv2d(64, 64, 1, same_padding=False, bn=bn))
        self.parallel_2 = nn.Sequential(Dilated_Conv2D(64, 64, 3, dilation=4, bn=bn))
        self.parallel_3 = nn.Sequential(Dilated_Conv2D(64, 64, 3, dilation=8, bn=bn))
        self.parallel_4 = nn.Sequential(Dilated_Conv2D(64, 64, 3, dilation=12, bn=bn))

        self.fuse = nn.Sequential(Conv2d(128, 1, 1, same_padding=False, bn=bn))

        self.fuse_1 = nn.Sequential(Conv2d(512, 384, 1, same_padding=False, bn=bn),
                                    Conv2d(384, 256, 3, same_padding=True, bn=bn),
                                    Conv2d(256, 128, 3, same_padding=True, bn=bn))

        # self.image_pool = nn.AdaptiveMaxPool2d(1)
        self.image_pool = nn.AdaptiveAvgPool2d(1)
        self.image_pool_conv = nn.Sequential(Conv2d(64, 64, 1, same_padding=False, bn=bn))
        self.conv_b1 = nn.Sequential(Conv2d(64, 64, 1, same_padding=False, bn=bn))
        self.conv_b2 = nn.Sequential(Conv2d(64, 64, 1, same_padding=False, bn=bn))
        self.conv_b3 = nn.Sequential(Conv2d(64, 64, 1, same_padding=False, bn=bn))
        self.conv_pipe = nn.Sequential(Conv2d(320, 320, 1, same_padding=False, bn=bn))



    def forward(self, im_data):

        x_1 = self.multi_scale_1(im_data)
        x_2 = self.multi_scale_2(x_1)
        x_3 = self.multi_scale_3(x_2)
        b, c, h, w = x_3.size()

        # using adaptive max/avg pool
        # x = self.conv_pipe(torch.cat((self.parallel_1(x_3), self.parallel_2(x_3), self.parallel_3(x_3),\
        #                               self.parallel_4(x_3), self.image_pool_conv(self.image_pool(x_3).repeat(1, 1,\
        #                               x_3.size()[2], x_3.size()[3]))), 1))

        x = self.conv_pipe(torch.cat((self.parallel_1(x_3), self.parallel_2(x_3), self.parallel_3(x_3), \
                                      self.parallel_4(x_3),
                                      self.image_pool_conv(self.image_pool(x_3)).repeat(1,1,h,w)), 1))

        x = self.fuse(self.fuse_1(torch.cat((self.conv_b1(x_1), self.conv_b2(x_2), self.conv_b3(x_3), x), 1)))

        return x
