from torch.utils.data import Dataset
import torch
import os
import cv2
from PIL import Image
from torchvision import transforms
import numpy as np
import time
import random
import json
# from extract_sequences import extract_seq
#from config import *
# import edge_utils as edge_utils
# import label_utils as label_utils
from torch.utils.data import DataLoader

import torch.nn as nn
import torch
from axial_attention import AxialAttention
import numpy

import numpy as np

import ConvLSTM_model as lstm_model   
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self,hidden_channel1 ,
                      hidden_channel2 ,
                      hidden_channel3 ,
                      input_channel):
        super(Encoder, self).__init__()
        self.hidden1 = hidden_channel1
        self.hidden2 = hidden_channel2
        self.hidden3 = hidden_channel3
        self.con_layer1 = nn.Conv2d(in_channels=input_channel,out_channels=self.hidden1,
                                    kernel_size=4,stride=2,padding=1)
        self.gru_layer1 = lstm_model.W_ConvLSTM(hidden=self.hidden1, conv_kernel_size=5, input_channle=self.hidden1,
                                              )

        self.con_layer2 = nn.Conv2d(in_channels=self.hidden1, out_channels=self.hidden2,
                                    kernel_size=4, stride=2, padding=1)
        self.gru_layer2 = lstm_model.W_ConvLSTM(hidden=self.hidden2, conv_kernel_size=5, input_channle=self.hidden2,
                                              )

        self.con_layer3 = nn.Conv2d(in_channels=self.hidden2, out_channels=self.hidden3,
                                    kernel_size=4, stride=2, padding=1)
        self.gru_layer3 = lstm_model.W_ConvLSTM(hidden=self.hidden3, conv_kernel_size=5, input_channle=self.hidden3,
                                              )
    def forward(self, input):
        H = input.size()[-2]
        W = input.size()[-1]
        decode_lstm_input = []
        #grulayer_1&convlayer_1
        hidden_size1 = (input.size()[0], self.hidden1, int(H/2), int(W/2))
        h1 = torch.zeros(hidden_size1).cuda()
        c1 = torch.zeros(hidden_size1).cuda()
        state1 = (h1,c1)
        n_step = input.size()[1]
        layer2_input = []
        for i in range(n_step):
            x_t = input[:, i, :, :, :]
            x_t = self.con_layer1(x_t)
            h1,c1 = self.gru_layer1.forward(x_t, state1)
            state1 = (h1,c1)
            layer2_input.append(h1)
        decode_lstm_input.append(state1)
        # grulayer_2&convlayer_2
        hidden_size2 = (input.size()[0], self.hidden2, int(H / 4), int(W / 4))
        h2 = torch.zeros(hidden_size2).cuda()
        c2 = torch.zeros(hidden_size2).cuda()
        state2 = (h2,c2)
        layer3_input = []
        for i in range(n_step):
            x_t = layer2_input[i]
            x_t = self.con_layer2(x_t)
            h2,c2 = self.gru_layer2.forward(x_t, state2)
            state2= (h2,c2)
            layer3_input.append(h2)
        decode_lstm_input.append(state2)
        # grulayer_3&convlayer_3
        hidden_size3 = (input.size()[0], self.hidden3, int(H / 8), int(W / 8))
        h3 = torch.zeros(hidden_size3).cuda()
        c3 = torch.zeros(hidden_size3).cuda()
        state3 = (h3, c3)
        for i in range(n_step):
            x_t = layer3_input[i]
            x_t = self.con_layer3(x_t)
            h3, c3 = self.gru_layer3.forward(x_t, state3)
            state3 = (h3, c3)
        decode_lstm_input.append(state3)
        return decode_lstm_input

class Decoder(nn.Module):
    def __init__(self,hidden_channel1,
                      hidden_channel2,
                      hidden_channel3,
                      out_channel,
                      ):
        super(Decoder, self).__init__()
#         self.out_len = out_len

#         self.gru_layer3 = lstm_model.W_ConvLSTM(hidden=hidden_channel3, conv_kernel_size=5, input_channle=hidden_channel3,
#                                              )
        self.conv_layer3 = nn.ConvTranspose2d(in_channels=hidden_channel3, out_channels=hidden_channel2,
                                              kernel_size=4, stride=2, padding=1)
#         self.gru_layer2 = lstm_model.W_ConvLSTM(hidden=hidden_channel2, conv_kernel_size=5, input_channle=hidden_channel2,
#                                               )
        self.conv_layer2 = nn.ConvTranspose2d(in_channels=hidden_channel2, out_channels=hidden_channel1,
                                              kernel_size=4, stride=2, padding=1)
#         self.gru_layer1 = lstm_model.W_ConvLSTM(hidden=hidden_channel1, conv_kernel_size=5, input_channle=hidden_channel1,
#                                               )
        self.conv_layer1 = nn.ConvTranspose2d(in_channels=hidden_channel1, out_channels=out_channel,
                                              kernel_size=4, stride=2, padding=1)
        
#         self.attn3 = AxialAttention(
#         dim = hidden_channel3,               # embedding dimension
#         dim_index = 1,         # where is the embedding dimension
#         #dim_heads = 300,        # dimension of each head. defaults to dim // heads if not supplied
#         heads = 8,             # number of heads for multi-head attention
#         num_dimensions = 2,    # number of axial dimensions (images is 2, video is 3, or more)
#         sum_axial_out = True   # whether to sum the contributions of attention on each axis, or to run the input through them sequentially. defaults to true
#         )
        
#         self.attn2 = AxialAttention(
#         dim = hidden_channel2,               # embedding dimension
#         dim_index = 1,         # where is the embedding dimension
#         #dim_heads = 300,        # dimension of each head. defaults to dim // heads if not supplied
#         heads = 8,             # number of heads for multi-head attention
#         num_dimensions = 2,    # number of axial dimensions (images is 2, video is 3, or more)
#         sum_axial_out = True   # whether to sum the contributions of attention on each axis, or to run the input through them sequentially. defaults to true
#         )
        
#         self.attn1 = AxialAttention(
#         dim = hidden_channel1,               # embedding dimension
#         dim_index = 1,         # where is the embedding dimension
#         #dim_heads = 300,        # dimension of each head. defaults to dim // heads if not supplied
#         heads = 8,             # number of heads for multi-head attention
#         num_dimensions = 2,    # number of axial dimensions (images is 2, video is 3, or more)
#         sum_axial_out = True   # whether to sum the contributions of attention on each axis, or to run the input through them sequentially. defaults to true
#         )
        
    def forward(self, states):
#         h3, c3 = states[-1]
#         state3 = states[-1]
#         layer2_input = []
#         for i in range(self.out_len):
#             paper_t = torch.zeros(h3.size()[0],  h3.size()[1],
#                                  h3.size()[-2], h3.size()[-1])
#             paper_t = paper_t.cuda()
#             h3,c3 = self.gru_layer3.forward(x_t=paper_t, hidden=state3)
#             state3 = (h3,c3)
#             hidden3_t = self.conv_layer3(h3)
#             layer2_input.append(hidden3_t)
#         state2 = states[-2]
#         layer1_input = []
#         for i in range(self.out_len):
#             h2,c2 = self.gru_layer2.forward(x_t=layer2_input[i], hidden=state2)
#             state2 = (h2,c2)
#             hidden2_t = self.conv_layer2(h2)
#             layer1_input.append(hidden2_t)
#         state1 = states[0]
#         out = []
#         for i in range(self.out_len):
#             h1, c1 = self.gru_layer1.forward(x_t=layer1_input[i], hidden=state1)
#             state1 = (h1, c1)
#             hidden1_t = self.conv_layer1(h1)
#             out.append(hidden1_t)
#         outs = torch.stack(out, 1)
        states = torch.squeeze(states, 1)
#         print(states.shape)
#         states = self.attn3(states)
#         print(states.shape)
        x = self.conv_layer3(states)
#         print(x.shape)
#         x = self.attn2(x)
        x = self.conv_layer2(x)
#         x = self.attn1(x)
        outs = self.conv_layer1(x)
        return outs
        
class Gen1(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim1, hidden_dim2, hidden_dim3):
        super(Gen1, self).__init__()
#         self.TemporalEncoder = TemporalEncoder(t_length, input_dim, hidden_dim, wide)
        self.encoder = Encoder(hidden_channel1=hidden_dim1, hidden_channel2=hidden_dim2,
                                       hidden_channel3=hidden_dim3, input_channel=input_dim)
        self.decoder = Decoder(hidden_channel1=hidden_dim1, hidden_channel2=hidden_dim2,
                                       hidden_channel3=hidden_dim3, out_channel=output_dim)
#         self.patch_size = patch_size
        self.attn = AxialAttention(
        dim = hidden_dim3,               # embedding dimension
        dim_index = 1,         # where is the embedding dimension
        #dim_heads = 300,        # dimension of each head. defaults to dim // heads if not supplied
        heads = 16,             # number of heads for multi-head attention
        num_dimensions = 2,    # number of axial dimensions (images is 2, video is 3, or more)
        sum_axial_out = True   # whether to sum the contributions of attention on each axis, or to run the input through them sequentially. defaults to true
    )
#         self.conv = nn.Conv2d(in_channels=hidden_dim,
#                               out_channels=patch_size*patch_size,
#                               kernel_size=(3, 3),
#                               stride=(1, 1),
#                               padding=(3 // 2, 3 // 2),
#                               bias=True)


    def forward(self, x):
#         print(x.shape)
#         x = reshape_patch(x, self.patch_size)
        x = self.encoder(x)[-1][0]
#         print(x.shape)
        x = self.attn(x)
        x = self.decoder(x)
#         x = reshape_patch_back(x, self.patch_size)
#         print(x.shape)
#         x = self.conv(x)
#         print(x.shape)
#         x = torch.unsqueeze(x, 1)
#         x = reshape_patch_back(x, self.patch_size)
        
#         x = torch.squeeze(x, 1)
#         x = self.attn(x)
#         return torch.sigmoid(x)
        return x

        # output = model(input_datas)[:,-1,:,:,:]
        # output = attn(output)
        
class Gen2(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim1, hidden_dim2, hidden_dim3):
        super(Gen2, self).__init__()
#         self.TemporalEncoder = TemporalEncoder(t_length, input_dim, hidden_dim, wide)
        self.encoder = Encoder(hidden_channel1=hidden_dim1, hidden_channel2=hidden_dim2,
                                       hidden_channel3=hidden_dim3, input_channel=input_dim)
        self.decoder = Decoder(hidden_channel1=hidden_dim1, hidden_channel2=hidden_dim2,
                                       hidden_channel3=hidden_dim3, out_channel=output_dim)
#         self.patch_size = patch_size
        self.attn = AxialAttention(
        dim = hidden_dim3,               # embedding dimension
        dim_index = 1,         # where is the embedding dimension
        #dim_heads = 300,        # dimension of each head. defaults to dim // heads if not supplied
        heads = 16,             # number of heads for multi-head attention
        num_dimensions = 2,    # number of axial dimensions (images is 2, video is 3, or more)
        sum_axial_out = True   # whether to sum the contributions of attention on each axis, or to run the input through them sequentially. defaults to true
    )
#         self.conv = nn.Conv2d(in_channels=hidden_dim,
#                               out_channels=patch_size*patch_size,
#                               kernel_size=(3, 3),
#                               stride=(1, 1),
#                               padding=(3 // 2, 3 // 2),
#                               bias=True)


    def forward(self, x):
#         print(x.shape)
#         x = reshape_patch(x, self.patch_size)
        x = self.encoder(x)[-1][0]
#         print(x.shape)
        x = self.attn(x)
        x = self.decoder(x)
#         x = reshape_patch_back(x, self.patch_size)
#         print(x.shape)
#         x = self.conv(x)
#         print(x.shape)
#         x = torch.unsqueeze(x, 1)
#         x = reshape_patch_back(x, self.patch_size)
        
#         x = torch.squeeze(x, 1)
#         x = self.attn(x)
#         return torch.sigmoid(x)
        return x

        # output = model(input_datas)[:,-1,:,:,:]
        # output = attn(output)

class Gen3(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim1, hidden_dim2, hidden_dim3):
        super(Gen3, self).__init__()
#         self.TemporalEncoder = TemporalEncoder(t_length, input_dim, hidden_dim, wide)
        self.encoder = Encoder(hidden_channel1=hidden_dim1, hidden_channel2=hidden_dim2,
                                       hidden_channel3=hidden_dim3, input_channel=input_dim)
        self.decoder = Decoder(hidden_channel1=hidden_dim1, hidden_channel2=hidden_dim2,
                                       hidden_channel3=hidden_dim3, out_channel=output_dim)
#         self.patch_size = patch_size
        self.attn = AxialAttention(
        dim = hidden_dim3,               # embedding dimension
        dim_index = 1,         # where is the embedding dimension
        #dim_heads = 300,        # dimension of each head. defaults to dim // heads if not supplied
        heads = 16,             # number of heads for multi-head attention
        num_dimensions = 2,    # number of axial dimensions (images is 2, video is 3, or more)
        sum_axial_out = True   # whether to sum the contributions of attention on each axis, or to run the input through them sequentially. defaults to true
    )
#         self.conv = nn.Conv2d(in_channels=hidden_dim,
#                               out_channels=patch_size*patch_size,
#                               kernel_size=(3, 3),
#                               stride=(1, 1),
#                               padding=(3 // 2, 3 // 2),
#                               bias=True)


    def forward(self, x):
#         print(x.shape)
#         x = reshape_patch(x, self.patch_size)
        x = self.encoder(x)[-1][0]
#         print(x.shape)
        x = self.attn(x)
        x = self.decoder(x)
#         x = reshape_patch_back(x, self.patch_size)
#         print(x.shape)
#         x = self.conv(x)
#         print(x.shape)
#         x = torch.unsqueeze(x, 1)
#         x = reshape_patch_back(x, self.patch_size)
        
#         x = torch.squeeze(x, 1)
#         x = self.attn(x)
#         return torch.sigmoid(x)
        return x

        # output = model(input_datas)[:,-1,:,:,:]
        # output = attn(output)

class Gen4(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim1, hidden_dim2, hidden_dim3):
        super(Gen4, self).__init__()
#         self.TemporalEncoder = TemporalEncoder(t_length, input_dim, hidden_dim, wide)
        self.encoder = Encoder(hidden_channel1=hidden_dim1, hidden_channel2=hidden_dim2,
                                       hidden_channel3=hidden_dim3, input_channel=input_dim)
        self.decoder = Decoder(hidden_channel1=hidden_dim1, hidden_channel2=hidden_dim2,
                                       hidden_channel3=hidden_dim3, out_channel=output_dim)
#         self.patch_size = patch_size
        self.attn = AxialAttention(
        dim = hidden_dim3,               # embedding dimension
        dim_index = 1,         # where is the embedding dimension
        #dim_heads = 300,        # dimension of each head. defaults to dim // heads if not supplied
        heads = 16,             # number of heads for multi-head attention
        num_dimensions = 2,    # number of axial dimensions (images is 2, video is 3, or more)
        sum_axial_out = True   # whether to sum the contributions of attention on each axis, or to run the input through them sequentially. defaults to true
    )
#         self.conv = nn.Conv2d(in_channels=hidden_dim,
#                               out_channels=patch_size*patch_size,
#                               kernel_size=(3, 3),
#                               stride=(1, 1),
#                               padding=(3 // 2, 3 // 2),
#                               bias=True)


    def forward(self, x):
#         print(x.shape)
#         x = reshape_patch(x, self.patch_size)
        x = self.encoder(x)[-1][0]
#         print(x.shape)
        x = self.attn(x)
        x = self.decoder(x)
#         x = reshape_patch_back(x, self.patch_size)
#         print(x.shape)
#         x = self.conv(x)
#         print(x.shape)
#         x = torch.unsqueeze(x, 1)
#         x = reshape_patch_back(x, self.patch_size)
        
#         x = torch.squeeze(x, 1)
#         x = self.attn(x)
#         return torch.sigmoid(x)
        return x

        # output = model(input_datas)[:,-1,:,:,:]
        # output = attn(output)


class VideoGANGenerator(nn.Module):

    def __init__(self):
        super(VideoGANGenerator, self).__init__()

        self.up1 = nn.ConvTranspose2d(
            1, 1, 4, stride=2, padding=1
        )
        self.up2 = nn.ConvTranspose2d(
            1, 1, 4, stride=2, padding=1
        )
        self.up3 = nn.ConvTranspose2d(
            1, 1, 4, stride=2, padding=1
        )

        # Generator #1
        input_dim = 17
        output_dim = 1
        hidden_dim1 = 32
        hidden_dim2 = 64
        hidden_dim3 = 128
        self.g1 = Gen1(input_dim, output_dim, hidden_dim1, hidden_dim2, hidden_dim3)
        self.g2 = Gen2(input_dim+1, output_dim, hidden_dim1, hidden_dim2, hidden_dim3)
        self.g3 = Gen3(input_dim+1, output_dim, hidden_dim1, hidden_dim2, hidden_dim3)
        self.g4 = Gen4(input_dim+1, output_dim, hidden_dim1, hidden_dim2, hidden_dim3)

    def forward(self, x):
        out = x
        h, w = x.shape[-2:]
        # print('x.shape is', x.shape)

        # TODO: Change the image size
        img1 = F.interpolate(out, size=(x.shape[2], int(h / 8), int(w / 8)))
        img2 = F.interpolate(out, size=(x.shape[2], int(h / 4), int(w / 4)))
        img3 = F.interpolate(out, size=(x.shape[2], int(h / 2), int(w / 2)))
        img4 = out
        # print(out.shape)
        # print(img1.shape)
        out = self.g1(img1)
        # print(out.shape)
        upsample1 = self.up1(out)

        # print(upsample1.shape)
        out = upsample1 + self.g2(torch.cat([upsample1.unsqueeze(1).expand(-1, 8, -1, -1, -1), img2], dim=2))
#         print(out.shape)
        upsample2 = self.up2(out)
        out = upsample2 + self.g3(torch.cat([upsample2.unsqueeze(1).expand(-1, 8, -1, -1, -1), img3], dim=2))
        upsample3 = self.up3(out)
        out = upsample3 + self.g4(torch.cat([upsample3.unsqueeze(1).expand(-1, 8, -1, -1, -1), img4], dim=2))

        # Apply tanh at the end
        out = torch.tanh(out)

        return out