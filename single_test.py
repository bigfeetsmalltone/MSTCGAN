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
#单独对测试集来进行划分

import torch.nn as nn
import torch
from axial_attention import AxialAttention
import numpy

import numpy as np
# numpy的形式
def reshape_patch(img_tensor, patch_size):
    assert 5 == img_tensor.ndim
    batch_size = np.shape(img_tensor)[0]
    seq_length = np.shape(img_tensor)[1]
    num_channels = np.shape(img_tensor)[2]
    img_height = np.shape(img_tensor)[3]
    img_width = np.shape(img_tensor)[4]
    a = np.reshape(img_tensor, [batch_size, seq_length, num_channels,
                                img_height//patch_size, patch_size,
                                img_width//patch_size, patch_size
                                ])
    b = np.transpose(a, [0,1,2,4,6,3,5])
    patch_tensor = np.reshape(b, [batch_size, seq_length, patch_size*patch_size*num_channels,
                                  img_height//patch_size,
                                  img_width//patch_size])
    return patch_tensor

# tensor的形式
def reshape_patch_back(patch_tensor, patch_size):
    assert 5 == patch_tensor.ndim
    batch_size = patch_tensor.shape[0]
    seq_length = patch_tensor.shape[1]
    channels = patch_tensor.shape[2]
    patch_height = patch_tensor.shape[3]
    patch_width = patch_tensor.shape[4]
    
    img_channels = channels // (patch_size*patch_size)
    a = torch.reshape(patch_tensor, [batch_size, seq_length,
                                  img_channels, patch_size, patch_size,
                                  patch_height, patch_width
                                  ])
    b = a.permute([0,1,2,5,4,3,6])
    img_tensor = torch.reshape(b, [batch_size, seq_length,
                                img_channels,
                                patch_height * patch_size,
                                patch_width * patch_size
                                ])
    return img_tensor
    

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
    """This class implements the full VideoGAN Generator Network.
    Currently a placeholder that copies the Vanilla GAN Generator network
    """

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
        
import torch.nn as nn

class TestSeqDataset(Dataset):

    # use the image pairs of first year as training data: len_tr = (1 + 365) * 365 / 2 = 66795
    # select some image pairs as valid data

    # datasets这是一个pytorch定义的dataset的源码集合。下面是一个自定义Datasets的基本框架，初始化放在__init__()中，
    # 其中__getitem__() 和__len__() 两个方法是必须重写的。__getitem__() 返回训练数据，如图片和label，
    # 而__len__() 返回数据长度。
    def __init__(self, image_dir, train_seq_len, pred_seq_len, mode, seed=2021):
        '''
        # image目录结构： image_dir/年月/时间.png
        # label目录结构： label_dir/self.mode/年月/时间.png
        '''

        self.seed = seed
        self.image_dir = image_dir
        #self.label_dir = label_dir
        self.train_seq_len = train_seq_len
        self.pred_seq_len = pred_seq_len
        self.mode = mode
#         self.pred_len = pred_len
#         self.seq_len = seq_len
        self.image_list = self.read_image_list(image_dir, mode, seed)
        self.len = len(self.image_list)
        print(self.image_list[-1], self.len)
        # self.crop_size = crop_size

        # self.train_transform = [
        #     # transforms.RandomCrop(448),
        #     transforms.RandomVerticalFlip(),
        #     transforms.RandomHorizontalFlip(),
        #     # transforms.RandomRotation(90),
        # ]

    def __getitem__(self, i):
        index = i // self.pred_seq_len 

        img, label = self.load_data(self.image_dir, self.image_list[index], i % self.pred_seq_len, self.mode) #这两行代码和之前不一样，需要注意一下
        #img = img / 255
        img[:, 0, :, :] = img[:, 0, :, :] / 255 #one-hot编码的维度不能进行归一化
        label = label / 255
        img = self.data_preproccess(img)
        label = self.data_preproccess(label)

        return img, label
        # return img.cuda(), label.cuda()

    def __len__(self):
        return self.len*self.pred_seq_len

    # 未划分数据集
    def read_image_list(self, image_dir, mode, seed):
        #对训练样本进行划分，按照训练集、验证集、测试集
        sequences = json.load(open(os.path.join(image_dir, 'sequences_48.json'), 'r'))
        
        # 取每个月的最后30%的序列做验证集
        labels_month = {}
        for month in range(201801, 201813):
             labels_month[str(month)] = []
        for seq in sequences:
            # print(seq)
            labels_month[seq[0][:6]].append(seq)
        labels_seperated = []
        for month in labels_month:
            len_seq_val = int(0.3 * len(labels_month[month]))
            if mode == 'train':
                labels_seperated.extend(labels_month[month][:-len_seq_val])
            elif mode == 'val':
                labels_seperated.extend(labels_month[month][-len_seq_val:-len_seq_val//2])
            else:
                labels_seperated.extend(labels_month[month][-len_seq_val//2:])

        return labels_seperated

    def load_data(self, image_dir, name_list, label_index, mode):
        '''
        加载数据
        :return:
        '''
        images = []
        #labels = []
        #num_classes = CHANNEL_OUT
        for idx, name in enumerate(name_list):
            temp = cv2.imread(os.path.join(image_dir, name[:6], name), cv2.IMREAD_GRAYSCALE)
            # temp = self.crop_image(temp, 256, 256)
            temp = self.new_crop_image(temp)
            images.append(temp)

#             if idx >= IN_LEN:
#                 label = cv2.imread(os.path.join(label_dir, name[:6], name[:29]+'.png'), cv2.IMREAD_GRAYSCALE)
#                 if(label is None): print(name)
#                 label = 1 * (label > 0)
#                 # 标注像素小于 n 的视为错标置零
#                 if np.sum(label) < LABEL_THRESHOLD:
#                     label = np.zeros(label.shape)
#                 labels.append(label)

#                 if W_EDGE > 0:
#                     # 生成类边界mask
#                     _edgemap = edge_utils.mask_to_onehot(label+1, num_classes)
#                     _edgemap = edge_utils.onehot_to_binary_edges(_edgemap, 2, num_classes)
#                     labels.append(_edgemap[0])
#                     # print(images[-1].shape, label.shape, _edgemap.shape)

#         merge = np.stack(images + labels, axis=0)
#         # merge = np.stack([image, light, label], axis=0)

#         # 去掉边界
#         r, c = merge[-1].shape  # 800 * 1280

#         # 中北部雷达范围外置零
#         merge[:, :MID_NORTH_ROW, :MID_NORTH_COL] = 0

#         # 去掉部分上下边界和左边雷达范围外区域
#         merge = merge[:, ROW_BOUNDARY:r-ROW_BOUNDARY, COL_BOUNDARY:]

#         if mode == 'train':
#         #     # random crop
#             z, r, c = merge.shape
#             # print(z,r,c)
#             crop_size = CROP_SIZE
            
#             np.random.seed(int(time.time()*1000000) % 2**32)
#             c_rand = int(np.random.randint(0, c-crop_size, 1))
#             if crop_size >= r:
#                 merge = merge[:, :, c_rand:c_rand+crop_size]
#             else:
#                 np.random.seed(int(time.time()*1000000) % 2**32)
#                 r_rand = int(np.random.randint(0, r-crop_size, 1))
#                 merge = merge[:, r_rand:r_rand+crop_size, c_rand:c_rand+crop_size]


#         #     # random rotate
#         #     np.random.seed(int(time.time()*1000000) % 2**32)
#         #     rot_k = int(np.random.randint(0, 4, 1))
#         #     merge = np.rot90(merge, rot_k, (1, 2))

#             # flip
#             np.random.seed(int(time.time()*1000000) % 2**32)
#             if int(np.random.randint(0, 2, 1)) == 1:
#                 merge = np.flip(merge, 1)
                
#             np.random.seed(int(time.time()*1000000) % 2**32)
#             if int(np.random.randint(0, 2, 1)) == 1:
#                 merge = np.flip(merge, 2)

#             # disturb value 0 ~ 255
#             # (1 - (260K-180K)/(320K-180K)) * 255 = 109.29
#             # (1-50/255)*(320K-180K) + 180K = 292.55
#             # 250K - 180K = 70K  70K/140K * 255 = 127.5
#             # val_min = np.min(merge[0,:,:])
#             val_min = CLIP_VAL  # 去掉小于60的像素，也就只是相当于将亮温高于285K的去掉了，对于对流是没有影响的
#             val_max = np.max(merge[0,:,:])
#             # print(val_min, val_max)
#             val_shift = 0  
#             np.random.seed(int(time.time()*1000000) % 2**32)
#             if int(np.random.randint(0, 2, 1)) == 1:
#                 val_shift = np.random.randint(-val_min, 0)
#             elif (255 - val_max) > 0:
#                 val_shift = np.random.randint(0, 255 - val_max)
#             # else:
#             #     # 就算将190K以上的都置为180K，应该影响也不大, 这个得再考虑考虑
#             #     val_shift = np.random.randint(0, 20)

#             merge[:len(name_list),:,:] = np.clip(np.array(merge[:len(name_list),:,:], dtype=int) + val_shift, 0, 255)

#         # print(image.shape)
#         image = (merge[:len(name_list),:,:] / 255.0 - 0.5) * 2  # 转到[-1, 1]
#         label = merge[len(name_list):,:,:]
        label = images[self.train_seq_len + label_index] #label_index从0开始
        image = images[0:self.train_seq_len]
        image = np.array(image)
        label = np.array(label)
        t, h, w = image.shape
        c = 1
        image = np.reshape(image, (t, c, h, w))
        image = self.add_seq_encoding(image, label_index)
#         if W_EDGE > 0:
#             c = 2
#         label = np.reshape(label, (t-IN_LEN, c, h, w))  # label and edge
#         label[:, 0] = label[:, 0] + 1
        # cv2.imwrite('label.png', (label[0, 0]-1) * 255)
        # cv2.imwrite('edge.png', (label[0, 1]*255))
        # print('save image')

        #return image.copy(), np.array(label, dtype=int)
        return image.copy(), label.copy()
    
    #剪切图像
    def crop_image(self, image, height_crop_size, width_crop_size):
        h, w = image.shape
        h_mid = h // 2
        w_mid = w // 2
        ans = image[h_mid-height_crop_size:h_mid+height_crop_size]
        ans = ans[:, w_mid-width_crop_size:w_mid+width_crop_size]
        return ans

    def new_crop_image(self, image):
        h, w = image.shape
        # h_mid = h // 2
        # w_mid = w // 2
        ans = image[32:768]
        # ans = ans[:, w_mid-width_crop_size:w_mid+width_crop_size]
        return ans
    
    #在通道维进行0-1编码
    def add_seq_encoding(self, image, label_index):
        t, c, h, w = image.shape
        temp = numpy.zeros((t, self.pred_seq_len, h, w))
        temp[:, label_index, :, :] = 1
        res = numpy.concatenate((image, temp), 1)
        return res
        
    def data_preproccess(self, data):
            '''
            数据预处理
            :param data:
            :return:
            '''
            # data = self.toTensor(data)
            data = torch.Tensor(data)
            return data

def constant_padding_image(image):
    # b, t, c, h, w = image.shape
    
    temp = image.squeeze(0)
    # print(temp.shape)
    # print(temp.shape)
    m = torch.nn.ReflectionPad2d((0, 0, 0, 32))
    return m(temp).unsqueeze(0)

def constant_padding_label(image):
    # b, t, c, h, w = image.shape
    
    temp = image.unsqueeze(0)
    # print(temp.shape)
    # print(temp.shape)
    m = torch.nn.ReflectionPad2d((0, 0, 0, 32))
    return m(temp).squeeze(0)
        
IMAGE_DIR_REAL = '/mnt/A/daikuai/LongSeqPredictionForConvection/convection_data/bright_images/'
train_seq_len = 8
pred_seq_len = 16

test_data = TestSeqDataset(IMAGE_DIR_REAL, train_seq_len, pred_seq_len, mode='test')
test_loader = DataLoader(dataset=test_data, batch_size=pred_seq_len, shuffle=False, num_workers=1)
#防止序列顺序被打乱
len(test_loader)

import os
from matplotlib.pyplot import imsave

def write_picture(dir, images):
    if not os.path.exists(dir):
        os.makedirs(dir)
    for idx, image in enumerate(images):
        cv2.imwrite(dir + str(1+idx) + '.png', image[3:3+730])

name = './model/G_temp_radar_model_15_16.ckpt'
temp_str = name.split('.')[1].split('_')
model_index = '-' + temp_str[-2] + '-' + temp_str[-1]
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def wrapper_test():
#     batch_size = 8
#     output_length = 30
#     input_length = 7
    # model = MetNet(input_length=input_length, output_length=output_length).cuda()
    
    model = torch.load(name)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    model = nn.DataParallel(model)
    model.eval()
#     print('model has been loaded')
#     test_data = Radar(
#         data_type='test',
#     )
#     test_loader = DataLoader(test_data,
#                               num_workers=0,
#                               batch_size=batch_size,
#                               shuffle=False,
#                               drop_last=False,
#                               pin_memory=False)
    with torch.no_grad():
        for i_batch, batch_data in enumerate(test_loader):
            seq_images, seq_labels = batch_data
            # temp_images = []
            # temp_labels = []
            for i in range(len(seq_images)):
                seq_image = seq_images[i].cuda().unsqueeze(0)
                seq_label = seq_labels[i].cuda().unsqueeze(0)

                seq_image = constant_padding_image(seq_image)
                seq_label = constant_padding_label(seq_label)
                
                pred_image = model(seq_image)
                pred_image = pred_image.squeeze(1)
                if i == 0:
                    temp_labels = (seq_label.detach().cpu().numpy()*255)
                    temp_images = (pred_image.detach().cpu().numpy()*255)
                else:
                    # print(temp_labels.shape)
                    # print(seq_label.shape)
                    temp_labels = np.concatenate((temp_labels, seq_label.detach().cpu().numpy()*255), 0)
                    temp_images = np.concatenate((temp_images, pred_image.detach().cpu().numpy()*255), 0)
        #         print(seq_labels.shape, pred_images.shape)
        #         seq_images = seq_images.detach().cpu().numpy()*255
            # seq_labels = seq_labels.detach().cpu().numpy()*255
            # pred_images = pred_images.detach().cpu().numpy()*255
            seq_labels = temp_labels
            pred_images = temp_images
            write_picture('./results' + model_index + '/' + str(i_batch) +  '/groud_truth/', seq_labels)
            write_picture('./results' + model_index + '/' + str(i_batch) + '/pred/', pred_images)
    #         break


wrapper_test()

