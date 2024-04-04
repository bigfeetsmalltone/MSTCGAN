import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math

class W_ConvLSTM(nn.Module):
    def __init__(self, hidden = 2, conv_kernel_size = 3,input_channle = 1):
        super(W_ConvLSTM, self).__init__()
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.hidden_channel = hidden

        self.conv_kernel_size = conv_kernel_size
        self.input_channel = input_channle
        self.padding = math.floor(conv_kernel_size / 2)
        self.build_model()
    def get_parameter(self,shape,init_method = 'xavier'):
        param = Parameter(torch.Tensor(*shape).cuda())
        if init_method == 'xavier':
            nn.init.xavier_uniform_(param)
        elif init_method == 'zero':
            nn.init.constant_(param,0)
        else:
            raise ('init method error')
        return param

    def build_model(self):

        input_to_state_shape = [
            self.hidden_channel,
            self.input_channel,
            self.conv_kernel_size,
            self.conv_kernel_size
        ]
        state_to_state_shape = [
            self.hidden_channel,
            self.hidden_channel,
            self.conv_kernel_size,
            self.conv_kernel_size
        ]
        state_bias_shape = [
            1, self.hidden_channel, 1, 1
        ]

        self.w_xi = self.get_parameter(input_to_state_shape)
        self.w_hi = self.get_parameter(state_to_state_shape)
        self.w_xf = self.get_parameter(input_to_state_shape)
        self.w_hf = self.get_parameter(state_to_state_shape)
        self.w_xc = self.get_parameter(input_to_state_shape)
        self.w_hc = self.get_parameter(state_to_state_shape)
        self.w_xo = self.get_parameter(input_to_state_shape)
        self.w_ho = self.get_parameter(state_to_state_shape)


        self.b_i = self.get_parameter(state_bias_shape,'zero')
        self.b_f = self.get_parameter(state_bias_shape,'zero')
        self.b_c = self.get_parameter(state_bias_shape,'zero')
        self.b_o = self.get_parameter(state_bias_shape, 'zero')
    def forward(self, x_t, hidden):
        h_tm1,c_tm1 = hidden
        i = self.sigmoid(
            F.conv2d(x_t, self.w_xi, bias=None, padding=self.padding)
            + F.conv2d(h_tm1, self.w_hi, bias=None, padding=self.padding)
            + self.b_i
        )

        f = self.sigmoid(
            F.conv2d(x_t, self.w_xf, bias=None, padding=self.padding)
            + F.conv2d(h_tm1, self.w_hf, bias=None, padding=self.padding)
            + self.b_f
        )

        c = f*c_tm1 + i*self.tanh(
            F.conv2d(x_t, self.w_xc, bias=None, padding=self.padding)
            + F.conv2d(h_tm1, self.w_hc, bias=None, padding=self.padding)
            + self.b_c
        )

        o = self.sigmoid(
            F.conv2d(x_t, self.w_xo, bias=None, padding=self.padding)
            + F.conv2d(h_tm1, self.w_ho, bias=None, padding=self.padding)
            + self.b_o
        )
        H = o*self.tanh(c)
        return H,c