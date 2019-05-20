import torch
import torch.nn as nn
from torch.autograd import Variable,Function
import numpy as np

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, bn=False):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Dilated_Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=True, bn=False, dilation=1):
        super(Dilated_Conv2D, self).__init__()
        padding = dilation if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Atrous_Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, rate=1):
        super(Atrous_Bottleneck, self).__init__()
        self.conv1 = Conv2d(inplanes, planes, kernel_size=1, same_padding=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Dilated_Conv2D(planes, planes, kernel_size=3, stride=stride, same_padding=True, dilation=rate)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = Conv2d(planes, planes, kernel_size=1, relu=False, same_padding=False)
        # self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv3(self.conv2(self.conv1(x)))
        out += residual
        out = self.relu(out)
        return out

class Deconv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=2, relu=True, same_padding=False, bn=False):
        super(Deconv2D, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

# class recursive_module(nn.Module):
#     def __init__(self,bn=False):
#         super(recursive_module, self).__init__()
#         self.recursive_model = nn.Sequential(Conv2d(4, 64, 3, same_padding=True, bn=bn),
#                                              Conv2d(64, 64, 3, same_padding=True, bn=bn),
#                                              Dilated_Conv2D(64, 64, 3, dilation=2, bn=bn),
#                                              Conv2d(64, 64, 3, same_padding=True, bn=bn),
#                                              Dilated_Conv2D(64, 64, 3, dilation=2, bn=bn),
#                                              Conv2d(64, 64, 3, same_padding=True, bn=bn),
#                                              Conv2d(64, 1, 1, same_padding=False, bn=bn))
#
#         self.T = 2
#
#     def forward(self,im_data,input,is_cuda=True):
#         x = input
#         for i in range(self.T):
#             x = torch.cat((im_data, x),1)
#             output = self.recursive_model(x)
#             x = output.detach()
#
#         if is_cuda:
#             x = x.cuda(2)
#
#         return x

#TODO:
class rbLSTM(nn.LSTM):
    def __init__(self):
        pass

    def forward(self):
        pass


class FC(nn.Module):
    def __init__(self, in_features, out_features, relu=True):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.fc(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


def save_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='w')
    for k, v in net.state_dict().items():
        h5f.create_dataset(k, data=v.cpu().numpy())


def load_net(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='r')
    for k, v in net.state_dict().items():        
        param = torch.from_numpy(np.asarray(h5f[k]))         
        v.copy_(param)


def load_modified_ms(fname, net):
    import h5py
    h5f = h5py.File(fname, mode='r')

    h5f_arr = np.asarray(h5f)

    for k,v in net.state_dict().items():
        # if k == 'DME.fuse_1.0.conv.bias' or k == 'DME.fuse_1.0.conv.weight' or k == 'DME.fuse.0.conv.bias'\
        #         or k == 'DME.fuse.0.conv.weight':
        # if k == 'DME.fuse_1.0.conv.bias' or k == 'DME.fuse_1.0.conv.weight' or k == 'DME.recursive.0.conv.weight' or \
        #     k == 'DME.recursive.0.conv.bias':
        # if k == 'DME.multi_scale_rec_1.0.conv.weight' or k == 'DME.multi_scale_rec_1.0.conv.bias':
        if k == 'image_pool_conv.0.conv.weight' or k == 'image_pool_conv.0.conv.bias':
            continue
        else:
            if k in h5f_arr:
                param = torch.from_numpy(np.asarray(h5f[k]))
            else:
                continue

        v.copy_(param)

def load_merged_mcnn_ms(fname_1, fname_2, net):
    import h5py
    h5f_1 = h5py.File(fname_1, mode='r')
    h5f_2 = h5py.File(fname_2, mode='r')

    h5f_1_arr = np.asarray(h5f_1)
    h5f_2_arr = np.asarray(h5f_2)

    for k,v in net.state_dict().items():
        # if k == 'DME.fuse.0.conv.bias' or k == 'DME.fuse.0.conv.weight':
        #    continue
        # else:
        #     if k in h5f_1_arr:
        #         param = torch.from_numpy(np.asarray(h5f_1[k]))
        #     elif k in h5f_2_arr:
        #         param = torch.from_numpy(np.asarray(h5f_2[k]))
        #     elif k in h5f_3_arr:
        #         param = torch.from_numpy(np.asarray(h5f_3[k]))

        if k == 'DME.fuse.0.conv.bias' or k == 'DME.fuse.0.conv.weight':
            continue
        else:
            if k in h5f_1_arr:
                param = torch.from_numpy(np.asarray(h5f_1[k]))
            elif k in h5f_2_arr:
                param = torch.from_numpy(np.asarray(h5f_2[k]))
            else:
                continue

        v.copy_(param)


def load_merged_three_net(fname_1, fname_2, fname_3, net):
    import h5py
    h5f_1 = h5py.File(fname_1, mode='r')
    h5f_2 = h5py.File(fname_2, mode='r')
    h5f_3 = h5py.File(fname_3, mode='r')

    h5f_1_arr = np.asarray(h5f_1)
    h5f_2_arr = np.asarray(h5f_2)
    h5f_3_arr = np.asarray(h5f_3)

    for k,v in net.state_dict().items():
        # if k == 'DME.fuse.0.conv.bias' or k == 'DME.fuse.0.conv.weight':
        #    continue
        # else:
        #     if k in h5f_1_arr:
        #         param = torch.from_numpy(np.asarray(h5f_1[k]))
        #     elif k in h5f_2_arr:
        #         param = torch.from_numpy(np.asarray(h5f_2[k]))
        #     elif k in h5f_3_arr:
        #         param = torch.from_numpy(np.asarray(h5f_3[k]))

        if k == 'DME.fuse.0.conv.bias' or k == 'DME.fuse.0.conv.weight':
            continue
        else:
            if k in h5f_1_arr:
                param = torch.from_numpy(np.asarray(h5f_1[k]))
            elif k in h5f_2_arr:
                param = torch.from_numpy(np.asarray(h5f_2[k]))
            elif k in h5f_3_arr:
                param = torch.from_numpy(np.asarray(h5f_3[k]))
            else:
                continue

        v.copy_(param)

def load_merged_four_net(fname_1, fname_2, fname_3, fname_4, net):
    import h5py
    h5f_1 = h5py.File(fname_1, mode='r')
    h5f_2 = h5py.File(fname_2, mode='r')
    h5f_3 = h5py.File(fname_3, mode='r')
    h5f_4 = h5py.File(fname_4, mode='r')

    h5f_1_arr = np.asarray(h5f_1)
    h5f_2_arr = np.asarray(h5f_2)
    h5f_3_arr = np.asarray(h5f_3)
    h5f_4_arr = np.asarray(h5f_4)

    # for k,v in net.state_dict().items():
    #     if k == 'DME.fuse.0.conv.bias' or k == 'DME.fuse.0.conv.weight':
    #         continue
    #     else:
    #         if k in h5f_1_arr:
    #             param = torch.from_numpy(np.asarray(h5f_1[k]))
    #         elif k in h5f_2_arr:
    #             param = torch.from_numpy(np.asarray(h5f_2[k]))
    #         elif k in h5f_3_arr:
    #             param = torch.from_numpy(np.asarray(h5f_3[k]))
    #         elif k in h5f_4_arr:
    #             param = torch.from_numpy(np.asarray(h5f_4[k]))
    #     v.copy_(param)

    for k,v in net.state_dict().items():
        if k == 'DME.fuse.0.conv.bias' or k == 'DME.fuse.0.conv.weight':
            continue
        else:
            if k in h5f_1_arr:
                param = torch.from_numpy(np.asarray(h5f_1[k]))
            elif k in h5f_2_arr:
                param = torch.from_numpy(np.asarray(h5f_2[k]))
            elif k in h5f_3_arr:
                param = torch.from_numpy(np.asarray(h5f_3[k]))
            elif k in h5f_4_arr:
                param = torch.from_numpy(np.asarray(h5f_4[k]))
            else:
                continue
        v.copy_(param)

def load_merged_all_net(fname_1, fname_2, fname_3, fname_4, fname_5, net):
    import h5py
    h5f_1 = h5py.File(fname_1, mode='r')
    h5f_2 = h5py.File(fname_2, mode='r')
    h5f_3 = h5py.File(fname_3, mode='r')
    h5f_4 = h5py.File(fname_4, mode='r')
    h5f_5 = h5py.File(fname_5, mode='r')

    h5f_1_arr = np.asarray(h5f_1)
    h5f_2_arr = np.asarray(h5f_2)
    h5f_3_arr = np.asarray(h5f_3)
    h5f_4_arr = np.asarray(h5f_4)
    h5f_5_arr = np.asarray(h5f_5)

    for k,v in net.state_dict().items():
        if k == 'DME.fuse.0.conv.bias' or k == 'DME.fuse.0.conv.weight':
            continue
        else:
            if k in h5f_1_arr:
                param = torch.from_numpy(np.asarray(h5f_1[k]))
            elif k in h5f_2_arr:
                param = torch.from_numpy(np.asarray(h5f_2[k]))
            elif k in h5f_3_arr:
                param = torch.from_numpy(np.asarray(h5f_3[k]))
            elif k in h5f_4_arr:
                param = torch.from_numpy(np.asarray(h5f_4[k]))
            elif k in h5f_5_arr:
                param = torch.from_numpy(np.asarray(h5f_5[k]))

        v.copy_(param)

def np_to_variable(x, is_cuda=True, is_training=False, dtype=torch.FloatTensor):
    v = torch.from_numpy(x)

    # gray image
    # v.unsqueeze_(0)
    # v.unsqueeze_(0)

    if len(v.shape) == 3:
        if v.shape[2] == 3:
            # rgb image
            v.unsqueeze_(0)
            v = v.permute(0, 3, 1, 2)
    elif len(v.shape) == 2:
        v.unsqueeze_(0)
        v.unsqueeze_(0)
    # print(v.type)

    if is_training:
        v = Variable(v.type(dtype))
    else:
        # v = Variable(v.type(dtype), requires_grad = False, volatile = True)
        v = Variable(v.type(dtype), requires_grad=False)
        # torch.set_grad_enabled(False)

    if is_cuda:
        v = v.cuda()
    return v

def tensor_to_cuda(x, is_cuda=True, is_training=False, dtype=torch.FloatTensor):

    # if is_training:
    #     v = Variable(v.type(dtype))
    # else:
    #     v = Variable(v.type(dtype), requires_grad=False)

    if is_cuda:
        x = x.cuda()
    return x

def np_to_variable_minibatch(x, is_training,is_cuda=True, is_density=False, dtype=torch.FloatTensor):
    ## Orginal One
    v = torch.from_numpy(x)

    if is_training:
        if is_density:
            # v.unsqueeze_(0)
            # v = v.permute(1, 0, 2, 3)
            v.unsqueeze_(1)
        else:
            v = v.permute(0, 3, 1, 2)
        v = Variable(v.type(dtype))
    else:
        if is_density:
            v.unsqueeze_(0)
            v.unsqueeze_(0)
        else:
            if len(v.shape) == 3:
                if v.shape[2] == 3:
                    v.unsqueeze_(0)
                    v = v.permute(0, 3, 1, 2)

        # v = Variable(v.type(dtype), requires_grad = False, volatile = True)
        v = Variable(v.type(dtype), requires_grad=False)
        # v = Variable(v.type(dtype))
    if is_cuda:
        v = v.cuda()
    return v

def set_trainable(model, requires_grad):
    for param in model.parameters():
        param.requires_grad = requires_grad


def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):                
                #print torch.sum(m.weight)
                m.weight.data.normal_(0.0, dev)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)

# def weights_normal_init_concat(model,dev=0.01)
#     for m in model:
#         if isinstance(m,)
#         weights_normal_init(m, dev)

def global_pool(x,is_cuda=True,is_training=False):
        # h_x = x.size()[2]
        # w_x = x.size()[3]
        if is_cuda:
            x = x.cuda()
        ave = nn.Sequential(nn.AvgPool2d((x.size()[2],x.size()[3])))
        x_global = ave(x)
        x_global = x_global.repeat(1,1,x.size()[2],x.size()[3])

        if is_training:
            x_global.requires_grid = True

        if is_cuda:
            x_global = x_global.cuda()
        return x_global


class MyReLU(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input
