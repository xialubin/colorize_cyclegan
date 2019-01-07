import numpy as np
from data import data_loader
import torch.nn as nn
import torch.nn.init as init
import torch
import functools
import itertools


def init_weight(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.weight.data, 0.0)
    print('initialize network with %s' % init_type)
    net.apply(init_func)


def Norm_layer(norm_type):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    # elif norm_type == 'none':
    #     norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def Padding(padding_type):
    if padding_type == 'reflect':
        padding = nn.ReflectionPad2d
    elif padding_type == 'replicate':
        padding = nn.ReflectionPad2d
    elif padding_type == 'zero':
        padding = nn.ZeroPad2d
    else:
        raise NotImplementedError('padding [%s] is not implemented' % padding_type)
    return padding


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)


#################
# generator
#################
class ResBlock(nn.Module):
    def __init__(self, channel, padding_type, norm='batch', dropout=True):
        super(ResBlock, self).__init__()
        norm_layer = Norm_layer(norm_type=norm)
        padding = Padding(padding_type=padding_type)

        conv_block = [padding(1),
                      nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, bias=True),
                      norm_layer(channel),
                      nn.ReLU(inplace=True)]  # inplace=True表示在输出变量的地址直接做出修改，而不是新建一个变量
        if dropout:
            conv_block += [nn.Dropout(0.5)]

        conv_block += [padding(1),
                       nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, bias=True),
                       norm_layer(channel)]
        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, inputs):
        return inputs + self.conv_block(inputs)


class GeneratorResNet(nn.Module):
    def __init__(self, input_nc, output_nc, padding_type, res_block=6, norm='batch', dropout=True):
        super(GeneratorResNet, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        norm_layer = Norm_layer(norm_type=norm)
        padding = Padding(padding_type)
        # input layer
        model = [padding(3),
                 nn.Conv2d(in_channels=input_nc, out_channels=64, kernel_size=7, bias=True),
                 norm_layer(64),
                 nn.ReLU(inplace=True)]
        # downsampling
        in_nc = 64
        out_nc = in_nc * 2
        for _ in range(2):
            model += [nn.Conv2d(in_channels=in_nc, out_channels=out_nc, kernel_size=3, stride=2, padding=1, bias=True),
                      norm_layer(out_nc),
                      nn.ReLU(inplace=True)]
            in_nc = out_nc
            out_nc = in_nc * 2
        # resblock
        for _ in range(res_block):
            model += [ResBlock(channel=in_nc, padding_type=padding_type, norm=norm, dropout=dropout)]
        # upsampling
        out_nc = in_nc // 2
        for _ in range(2):
            model += [nn.ConvTranspose2d(in_channels=in_nc, out_channels=out_nc, kernel_size=3, stride=2, padding=1, output_padding=1),
                      norm_layer(out_nc),
                      nn.ReLU(inplace=True)]
            in_nc = out_nc
            out_nc = in_nc // 2
        # output layer
        model += [padding(3),
                  nn.Conv2d(in_channels=in_nc, out_channels=output_nc, kernel_size=7),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, inputs):
        return self.model(inputs)


#################
# discriminator
#################
class Discriminator(nn.Module):
    def __init__(self, input_nc, output_nc, norm='batch'):
        super(Discriminator, self).__init__()
        norm_layer = Norm_layer(norm)

        def discriminatorblock(in_nc, out_nc, normalize=True):
            block = [nn.Conv2d(in_nc, out_nc, stride=2, kernel_size=4, padding=1)]
            if normalize:
                block += [norm_layer(out_nc)]
            block += [nn.LeakyReLU(0.2, inplace=True)]
            return block

        self.model = nn.Sequential(*discriminatorblock(input_nc, 64, normalize=False),
                                   *discriminatorblock(64, 128, normalize=True),
                                   *discriminatorblock(128, 256, normalize=True),
                                   *discriminatorblock(256, 512, normalize=True),
                                   nn.ZeroPad2d((1, 0, 1, 0)),
                                   nn.Conv2d(512, output_nc, kernel_size=4, padding=1))

    def forward(self, inputs):
        return self.model(inputs)


###################
# define loss and optimize
###################
class Model(object):
    def __init__(self, opt):
        # Losses
        self.criterion_GAN = torch.nn.MSELoss()
        self.criterion_cycle = torch.nn.L1Loss()
        self.criterion_identity = torch.nn.L1Loss()
        # model
        self.G_AB = GeneratorResNet(opt.input_nc, opt.output_nc, opt.padding_type, opt.res_num, opt.norm, opt.dropout)
        self.G_BA = GeneratorResNet(opt.output_nc, opt.input_nc, opt.padding_type, opt.res_num, opt.norm, opt.dropout)
        self.D_A = Discriminator(opt.input_nc, 1, opt.norm)
        self.D_B = Discriminator(opt.output_nc, 1, opt.norm)
        # to device
        if opt.device == 'gpu':
            device = torch.device("cuda:0")
            self.criterion_GAN.to(device)
            self.criterion_cycle.to(device)
            self.criterion_identity.to(device)
            self.G_AB.to(device)
            self.G_BA.to(device)
            self.D_A.to(device)
            self.D_B.to(device)

        if opt.isTrainning and (not opt.continue_train):
            # initialize models
            init_weight(self.G_AB, init_type=opt.initial)
            init_weight(self.G_BA, init_type=opt.initial)
            init_weight(self.D_A, init_type=opt.initial)
            init_weight(self.D_B, init_type=opt.initial)
        else:
            # Load pretrained models
            self.G_AB.load_state_dict(torch.load('%s/G_AB.pth' % opt.checkpoint))
            self.G_BA.load_state_dict(torch.load('%s/G_BA.pth' % opt.checkpoint))
            self.D_A.load_state_dict(torch.load('%s/D_A.pth' % opt.checkpoint))
            self.D_B.load_state_dict(torch.load('%s/D_B.pth' % opt.checkpoint))
        # optimize operator
        self.optimize_G = torch.optim.Adam(itertools.chain(self.G_AB.parameters(), self.G_BA.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))
        self.optimize_D_A = torch.optim.Adam(self.D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        self.optimize_D_B = torch.optim.Adam(self.D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
        # learning rate schedule
        self.lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(self.optimize_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
        self.lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(self.optimize_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
        self.lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(self.optimize_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

    def set_input(self):
        pass

    def forward(self):
        pass

    def backward_G(self):
        pass

    def backward_D_A(self):
        pass

    def backward_D_B(self):
        pass

    def optimize_params(self):
        pass
