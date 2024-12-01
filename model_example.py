import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class LinearBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearBlock, self).__init__()
        self.fc=nn.Linear(in_channels, out_channels)
        self.af=nn.LeakyReLU(0.2)

    def forward(self, x):
        x=self.fc(x)
        return self.af(x)

class ConvBlockSN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlockSN, self).__init__()
        self.conv_sn=spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
        #self.conv_sn=nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.af=nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.af(self.conv_sn(x))

class FrontEncoder(nn.Module):
    def __init__(self, in_channels=2):
        super(FrontEncoder, self).__init__()
        # img size:           128, 64, 32, 16,  8
        channel_list=[in_channels, 16, 32, 64, 64]
        kernel_list= [4, 4, 4, 4]
        stride_list= [2, 2, 2, 1]
        padding_list=[1, 1, 1, 0]
        self.encoder_conv_sn=nn.Sequential(*[
            ConvBlockSN(  channel_list[layer_no],
                          channel_list[layer_no+1],
                          kernel_list[layer_no],
                          stride_list[layer_no],
                          padding_list[layer_no])
            for layer_no in range(len(channel_list)-1)])

    def forward(self, x):
        return self.encoder_conv_sn(x)

class EndEncoder(nn.Module):
    def __init__(self, in_channels=40):
        super(EndEncoder, self).__init__()
        # img size:             8,  4,  2,  1
        channel_list=[in_channels, 32, 64,128]
        kernel_list =[           3,  3,  2]
        stride_list =[           2,  1,  1]
        padding_list=[           1,  0,  0]
        self.encoder_conv_sn=nn.Sequential(*[
            ConvBlockSN(  channel_list[layer_no],
                          channel_list[layer_no+1],
                          kernel_list[layer_no],
                          stride_list[layer_no],
                          padding_list[layer_no])
            for layer_no in range(len(channel_list)-1)])

    def forward(self, x):
        return self.encoder_conv_sn(x)

class SingleDisc(nn.Module):
    def __init__(self, in_channels=2):
        super(SingleDisc, self).__init__()
        self.front_encoder=FrontEncoder(in_channels)
        self.fc=nn.Conv2d(64, 1, kernel_size=4, stride=1, padding=0)

    def forward(self, x):
        return self.fc(self.front_encoder(x))

class DiscS(nn.Module):
    def __init__(self, in_channels=2):
        super(DiscS, self).__init__()
        self.disc_ang=SingleDisc(in_channels)
        self.disc_dop=SingleDisc(in_channels)

    def forward(self, sim, target):
        batch_size, batch_length,channel_size,height,width=sim.shape
        bl_size=batch_size*batch_length
        ang=torch.cat((sim[...,0:1,:,:],target[...,0:1,:,:]), -3)
        dop=torch.cat((sim[...,1:2,:,:],target[...,1:2,:,:]), -3)
        ang=self.disc_ang(ang.view(bl_size,channel_size,height,width))
        dop=self.disc_dop(dop.view(bl_size,channel_size,height,width))
        x=torch.cat((ang,dop), -3)
        return x

if __name__=='__main__':
    batch_size=13
    batch_length=7
    print('-----------------------------')
    print('LinearBlock:')
    x=torch.rand((batch_size*batch_length, 20))
    print('\tInput:', x.shape)
    model=LinearBlock(20, 10)
    x=model(x)
    print('\tOutput:', x.shape)
    print('-----------------------------')
    print('ConvBlockSN:')
    x=torch.rand((batch_size*batch_length, 3, 16, 16))
    print('\tInput:', x.shape)
    model=ConvBlockSN(in_channels=3,out_channels=8)
    x=model(x)
    print('\tOutput:', x.shape)
    print('-----------------------------')
    print('FrontEncoder:')
    x=torch.rand((batch_size*batch_length, 2, 128, 128))
    print('\tInput:', x.shape)
    model=FrontEncoder()
    x=model(x)
    print('\tOutput:', x.shape)
    print('-----------------------------')
    print('EndEncoder:')
    x=torch.rand((batch_size*batch_length, 40, 8, 8))
    print('\tInput:', x.shape)
    model=EndEncoder()
    x=model(x)
    print('\tOutput:', x.shape)
    print('-----------------------------')
    print('SingleDisc:')
    x=torch.rand((batch_size*batch_length, 2, 128, 128))
    print('\tInput:', x.shape)
    model=SingleDisc()
    x=model(x)
    print('\tOutput:', x.shape)
    print('-----------------------------')
    print('DiscS:')
    x=torch.rand((batch_size,batch_length, 2, 128, 128))
    print('\tInput:', x.shape, x.shape)
    model=DiscS()
    x=model(x,x)
    print('\tOutput:', x.shape)
    print('-----------------------------')
