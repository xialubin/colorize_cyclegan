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


class Discriminatorbuffer(object):
    def __init__(self, maxsize=50):
        self.maxsize = maxsize
        self.data = []

    def push_and_pop(self, inputs):
        to_return = []
        for elements in inputs.data:
            elements = torch.unsqueeze(elements, 0)
            if len(self.data) < self.maxsize:
                self.data.append(elements)
                to_return.append(elements)
            else:
                if np.random.uniform(0,1) > 0.5:
                    index = np.random.randint(low=0, high=self.maxsize)
                    to_return.append(self.data[index].clone())  # deep copy
                    self.data[index] = elements
                else:
                    to_return.append(elements)
        return torch.cat(to_return)


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
            self.device = device
            self.criterion_GAN.to(device)
            self.criterion_cycle.to(device)
            self.criterion_identity.to(device)
            self.G_AB.to(device)
            self.G_BA.to(device)
            self.D_A.to(device)
            self.D_B.to(device)

        if opt.isTraining and (not opt.continue_train):
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

        self.buffer_D_A = Discriminatorbuffer(maxsize=50)
        self.buffer_D_B = Discriminatorbuffer(maxsize=50)

        self.opt = opt

    def set_input(self, batch):
        self.real_A = batch['A'].to(self.device)
        self.real_B = batch['B'].to(self.device)

    def forward(self):
        self.fake_B = self.G_AB.forward(self.real_A)
        self.fake_A = self.G_BA.forward(self.real_B)
        self.ret_A = self.G_BA.forward(self.fake_B)
        self.ret_B = self.G_AB.forward(self.fake_A)

    def expand_target(self, shape, flags=True):
        if flags:
            return torch.ones(*shape).to(self.device)
        else:
            return torch.zeros(*shape).to(self.device)

    def backward_G(self):
        # gan loss
        score_AB = self.D_B.forward(self.fake_B)
        score_BA = self.D_A.forward(self.fake_A)
        if score_AB.shape != score_BA.shape:
            raise NotImplementedError("the shape of D(B) and D(A) should be same")
        shape = list(score_AB.shape)
        loss_G_AB = self.criterion_GAN(score_AB, self.expand_target(shape, True))
        loss_G_BA = self.criterion_GAN(score_BA, self.expand_target(shape, True))
        self.loss_GAN = (loss_G_AB + loss_G_BA) / 2.0

        # cycle loss
        loss_cycle_A = self.criterion_cycle(self.ret_A, self.real_A)
        loss_cycle_B = self.criterion_cycle(self.ret_B, self.real_B)
        self.loss_cycle = (loss_cycle_A + loss_cycle_B) / 2.0

        # total loss
        self.loss_G = self.loss_GAN + self.loss_cycle
        self.loss_G.backward()

    def backward_D_A(self):
        score_real = self.D_A.forward(self.real_A)
        fake_A_ = self.buffer_D_A.push_and_pop(self.fake_A)
        score_fake = self.D_A.forward(fake_A_.detach())
        if score_real.shape != score_fake.shape:
            raise NotImplementedError("the shape of D_A(real_img) and D_A(fake_img) should be same")
        shape = list(score_real.shape)
        loss_real = self.criterion_GAN(score_real, self.expand_target(shape, True))
        loss_fake = self.criterion_GAN(score_fake, self.expand_target(shape, False))

        self.loss_D_A = (loss_real + loss_fake) / 2.0
        self.loss_D_A.backward()

    def backward_D_B(self):
        score_real = self.D_B.forward(self.real_B)
        fake_B_ = self.buffer_D_B.push_and_pop(self.fake_B)
        score_fake = self.D_B.forward(fake_B_.detach())
        if score_real.shape != score_fake.shape:
            raise NotImplementedError("the shape of D_B(real_img) and D_B(fake_img) should be same")
        shape = list(score_real.shape)
        loss_real = self.criterion_GAN(score_real, self.expand_target(shape, True))
        loss_fake = self.criterion_GAN(score_fake, self.expand_target(shape, False))

        self.loss_D_B = (loss_real + loss_fake) / 2.0
        self.loss_D_B.backward()

    def optimize_params(self):
        # forward
        self.forward()
        # update G
        self.optimize_G.zero_grad()
        self.set_requires_grad([self.D_A, self.D_B], False)
        self.backward_G()
        self.optimize_G.step()

        # update D_A
        self.optimize_D_A.zero_grad()
        self.set_requires_grad([self.D_A, self.D_B], True)
        self.backward_D_A()
        self.optimize_D_A.step()
        # update D_B
        self.optimize_D_B.zero_grad()
        self.backward_D_B()
        self.optimize_D_B.step()

        # Update learning rates
        self.lr_scheduler_G.step()
        self.lr_scheduler_D_A.step()
        self.lr_scheduler_D_B.step()

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid computation"""
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def save_network(self):
        # save models to disk
        path = self.opt.checkpoint
        torch.save(self.G_AB, path + '/G_AB.pth')
        torch.save(self.G_BA, path + '/G_BA.pth')
        torch.save(self.D_A, path + '/D_A.pth')
        torch.save(self.D_B, path + '/D_B.pth')

    def sample_image(self):
        path = self.opt.checkpoint + '/sample/'
        fake_A_img = self.fake_A.to("cpu").data.numpy()[0, :, :, :].transpose(1, 2, 0)
        fake_B_img = self.fake_B.to("cpu").data.numpy()[0, :, :, :].transpose(1, 2, 0)
        ret_A_img = self.ret_A.to("cpu").data.numpy()[0, :, :, :].transpose(1, 2, 0)
        ret_B_img = self.ret_B.to("cpu").data.numpy()[0, :, :, :].transpose(1, 2, 0)
