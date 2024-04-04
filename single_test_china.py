from axial_attention import AxialAttention
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
from model import *

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

date_start = 201801
date_end = 201812

def extract_seq(file_list_all, interval = 15, error = 3, seq_len=5):
    file_time_list = []
    for idx, file_path in enumerate(file_list_all):
            # 将所有时间换算到时间戳的分钟数
            file_time = file_path[:14]
            date_time = time.strptime(file_time, '%Y%m%d%H%M%S')
            time_stamp = time.mktime(date_time)  # 秒数
            minutes = int(time_stamp // 60)
            file_time_list.append(minutes)
            # print(file_time, minutes)

    # 抽出来的以15+-3分钟为间隔, 8张图为一个序列
    sequences = []
    search_space = 48  # 四分钟一张图,则两小时内的图必在后36张图内
    for idx, t_first in enumerate(file_time_list):
        seq_i = [file_list_all[idx]]

        # 这种取 t_first + 15/30/45/... 前后3分钟内的图像比较合理,但一个序列都抽不出来
        idx_require = idx
        for i_t in range(1, seq_len):
            t_require = t_first + interval * i_t
            idx_other_start = idx_require + 1
            # print(t_require)
            for t_other in (file_time_list[idx_other_start:idx+search_space]):
                # print(t_other, t_require)
                idx_require += 1
                if np.abs(t_other - t_require) < error:
                    seq_i.append(file_list_all[idx_require])
                    break

        # # 这里直接判断前后两张的时间间隔是否15+-3分钟内, 一样抽不出来,要全圆盘的数据加进来
        # idx_require = idx
        # t_require = t_first
        # for i_t in range(1, 8):
        #     t_require = t_require + interval
        #     idx_other_start = idx_require + 1
        #     for t_other in (file_time_list[idx_other_start:idx+search_space]):
        #         idx_require += 1
        #         if np.abs(t_other - t_require) < error:
        #             seq_i.append(file_list_all[idx_require])
        #             t_require = t_other
        #             break
                    
        if len(seq_i) == seq_len:
            sequences.append(seq_i)

        # if idx % 1000 == 0:
        #     print(seq_i)
        
        # if idx > 100:
        # break
    print(sequences[1])
    print(len(sequences))
    return sequences


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
        # self.patch_size = patch_size
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
        # index = i // self.pred_seq_len

        img, label = self.load_data(self.image_dir, self.image_list[i], self.mode)
        img[:, 0:1, :, :] = img[:, 0:1, :, :] / 255 #one-hot编码的维度不能进行归一化
#         img[:, self.patch_size*self.patch_size, :, :] = img[:, self.patch_size*self.patch_size, :, :] / self.pred_seq_len
        label = label / 255
        img = self.data_preproccess(img)
        label = self.data_preproccess(label)

        return img, label
        # return img.cuda(), label.cuda()

    def __len__(self):
        return self.len


    def read_image_list(self, image_dir, mode, seed):
        file_list_all = []

        sequences = json.load(open(os.path.join(image_dir, 'sequences_24_china_202006.json'), 'r'))
        return sequences

    def load_data(self, image_dir, name_list, mode):
        '''
        加载数据
        :return:
        '''
        images = []
        #labels = []
        #num_classes = CHANNEL_OUT
        for idx, name in enumerate(name_list):
            print(os.path.join(image_dir, name[:6], name))
            # exit(0)
            temp = cv2.imread(os.path.join(image_dir, name[:6], name), cv2.IMREAD_GRAYSCALE)
            temp = temp.astype(np.float32) #将uint8转成float32
            # temp = self.crop_image(temp, 128, 128)
            images.append(temp)
        label = images[self.train_seq_len: self.train_seq_len+self.pred_seq_len] #label_index从0开始
        image = images[0:self.train_seq_len]
        image = np.array(image)
        label = np.array(label)
        t, h, w = image.shape
        c = 1
        b = 1
        image = np.reshape(image, (b, t, c, h, w))
        # image = reshape_patch(image, patch_size)
#         print(image.shape)
        image = numpy.squeeze(image, 0)
#         image = self.add_num_encoding(image, label_index)
        # image = self.add_seq_encoding(image, label_index) # 进行通道维的0-1编码
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
    
    #在通道维进行0-1编码
    def add_seq_encoding(self, image, label_index):
        t, c, h, w = image.shape
        temp = numpy.zeros((t, self.pred_seq_len, h, w))
        temp[:, label_index, :, :] = 1
        res = numpy.concatenate((image, temp), 1)
        return res
    
    #前面证实one-hot编码没有啥用，现在直接改为单通道的num编码。
    def add_num_encoding(self, image, label_index):
        t, c, h, w = image.shape
        temp = numpy.zeros((t, 1, h, w))
        temp[:, 0, :, :] = 1 + label_index
        res = numpy.concatenate((image, temp), 1)
        return res
        
    def data_preproccess(self, data):
            '''
            数据预处理
            :param data:
            :return:
            '''
            # data = self.toTensor(data)
#             print(data)
            data = torch.Tensor(data)
            return data

def constant_padding_image(image):
    # b, t, c, h, w = image.shape
    
    temp = image.squeeze(0)
    # print(temp.shape)
    # print(temp.shape)
    m = torch.nn.ReflectionPad2d((0, 0, 19, 19))
    return m(temp).unsqueeze(0)

def constant_padding_label(image):
    # b, t, c, h, w = image.shape
    
    temp = image.unsqueeze(0)
    # print(temp.shape)
    # print(temp.shape)
    # m = torch.nn.ReflectionPad2d((0, 0, 0, 32))
    m = torch.nn.ReflectionPad2d((0, 0, 19, 19))
    return m(temp).squeeze(0)
        
# IMAGE_DIR_REAL = '/mnt/A/daikuai/LongSeqPredictionForConvection/convection_data/bright_images/'
# data_path = '/mnt/A/south_east_all/'
data_path = './github_china_area_samples/'
train_seq_len = 8
pred_seq_len = 16

test_data = TestSeqDataset(data_path, train_seq_len, pred_seq_len, mode='test')
test_loader = DataLoader(dataset=test_data, batch_size=1, shuffle=False, num_workers=1)
#防止序列顺序被打乱
len(test_loader)

import os
from matplotlib.pyplot import imsave

def write_picture(dir, images):
    if not os.path.exists(dir):
        os.makedirs(dir)
    for idx, image in enumerate(images):
        # cv2.imwrite(dir + str(1+idx) + '.png', image)
        # imsave(dir + str(1+idx) + '.png', image, cmap="gray")
        print(image.shape)
        temp = image[19:19+730, :]
        # temp = temp[:, 21:21+1750]
        # print('this is a test')
        print(temp.shape)
        imsave(dir + str(1+idx) + '.png', temp, cmap="gray")
        # cv2.imwrite(dir + str(1+idx) + '.png', temp)

def write_picture_in(dir, images):
    if not os.path.exists(dir):
        os.makedirs(dir)
    for idx, image in enumerate(images):
        # cv2.imwrite(dir + str(1+idx) + '.png', image)
        imsave(dir + str(1+idx) + '.png', image, cmap="gray")
        # print(image.shape)
        # temp = image[18:18+1500, :]
        # temp = temp[:, 21:21+1750]
        # # print('this is a test')
        # print(temp.shape)
        # imsave(dir + str(1+idx) + '.png', temp, cmap="gray")

# name = './G_temp_radar_model_15_16.ckpt'
name ='./G_temp_radar_model_15_16.ckpt'
temp_str = name.split('.')[1].split('_')
model_index = '-' + temp_str[-2] + '-' + temp_str[-1]


def add_seq_encoding(image, label_index):
        t, c, h, w = image.shape
        temp = numpy.zeros((t, pred_seq_len, h, w))
        temp[:, label_index, :, :] = 1
        res = numpy.concatenate((image, temp), 1)
        res = torch.Tensor(res)
        return res

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
            # print(seq_images.shape)
            # print(seq_labels.shape)
            # temp_images = []
            # temp_labels = []
            for i in range(pred_seq_len):
                res = add_seq_encoding(seq_images[0], i)
                seq_image = res.cuda().unsqueeze(0)
                seq_label = seq_labels[0,i,:,:].cuda().unsqueeze(0)
                # print(seq_image.shape)
                seq_image = constant_padding_image(seq_image)
                seq_label = constant_padding_label(seq_label)
                # print(seq_image.shape)
                
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
            # print(seq_images.shape)
            # print(seq_labels.shape)
            write_picture('./results' + model_index + '_china/' + str(i_batch) +  '/groud_truth/', seq_labels)
            
            # pre_label = seq_images[0][-1][0].clone()
            # for i in range(len(pred_images)):
            #     pred_images[i] = hist_match_reverse(pred_images[i], pre_label.detach().cpu().numpy())

            write_picture('./results' + model_index + '_china/' + str(i_batch) + '/pred/', pred_images)
            write_picture_in('./results' + model_index + '_china/' + str(i_batch) + '/in/', seq_images[0,:,0,:,:].numpy()*255)

    #         break

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
wrapper_test()

