
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision
import os
import fusion_strategy


class activateF(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()

    def forward(self, x):
        return (F.softplus(2 * x) - 0.6931471805599453) / 2


class UpsampleReshape_eval(torch.nn.Module):
    def __init__(self):
        super(UpsampleReshape_eval, self).__init__()
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x1, x2):
        x2 = self.up(x2)
        shape_x1 = x1.size()
        shape_x2 = x2.size()
        left = 0
        right = 0
        top = 0
        bot = 0
        if shape_x1[3] != shape_x2[3]:
            lef_right = shape_x1[3] - shape_x2[3]
            if lef_right % 2 is 0.0:
                left = int(lef_right / 2)
                right = int(lef_right / 2)
            else:
                left = int(lef_right / 2)
                right = int(lef_right - left)

        if shape_x1[2] != shape_x2[2]:
            top_bot = shape_x1[2] - shape_x2[2]
            if top_bot % 2 is 0.0:
                top = int(top_bot / 2)
                bot = int(top_bot / 2)
            else:
                top = int(top_bot / 2)
                bot = int(top_bot - top)

        reflection_padding = [left, right, top, bot]
        reflection_pad = nn.ReflectionPad2d(reflection_padding)
        x2 = reflection_pad(x2)
        return x2



class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.dropout = nn.Dropout2d(p=0.5)  # ?
        self.act = activateF()
        self.is_last = is_last

    def forward(self, x):
        out = self.reflection_pad(x)

        out = self.act(self.conv2d(out))

        if self.is_last is False:
            out = F.relu(out, inplace=True)
        return out



class DenseBlock_light(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, img_size):
        super(DenseBlock_light, self).__init__()
        out_channels_def = int(in_channels / 2)

        denseblock = []
        denseblock += [ConvLayer(in_channels, out_channels_def, kernel_size, stride),
                       ConvLayer(out_channels_def, out_channels, 1, stride),
                       AxialBlock(out_channels, out_channels // 2, kernel_size=img_size)
                       ]

        self.denseblock = nn.Sequential(*denseblock)

    def forward(self, x):
        out = self.denseblock(x)
        return out


def imshow(img, name, type):

    img = torch.squeeze(img, dim=0)
    nrow = img.size(0)

    img = torchvision.utils.make_grid(img, nrow=nrow, normalize=True, padding=2)

    for i in range(nrow):
        path = './plt_png/' + str(name) + str(type)
        imgname = str(i) + '.jpg'
        if not os.path.exists(path):
            os.makedirs(path)

        torchvision.utils.save_image(img[i], os.path.join(path, imgname))



class AEFusion_autoencoder(nn.Module):
    def __init__(self, nb_filter, input_nc=1, output_nc=1, deepsupervision=True):
        super(AEFusion_autoencoder, self).__init__()
        self.deepsupervision = deepsupervision
        block = DenseBlock_light
        output_filter = 8  # 16
        kernel_size = 3
        stride = 1
        imgsize = [256, 128, 64, 32]
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2)
        self.up_eval = UpsampleReshape_eval()

        self.conv0 = ConvLayer(input_nc, output_filter, 1, stride)
        self.DB1_0 = block(output_filter, nb_filter[0], kernel_size, 1, imgsize[0])
        self.DB2_0 = block(nb_filter[0], nb_filter[1], kernel_size, 1, imgsize[1])
        self.DB3_0 = block(nb_filter[1], nb_filter[2], kernel_size, 1, imgsize[2])
        self.DB4_0 = block(nb_filter[2], nb_filter[3], kernel_size, 1, imgsize[3])

        self.DB1_1 = block(nb_filter[0] + nb_filter[1], nb_filter[0], kernel_size, 1, imgsize[0])
        self.DB2_1 = block(nb_filter[1] + nb_filter[2], nb_filter[1], kernel_size, 1, imgsize[1])
        self.DB3_1 = block(nb_filter[2] + nb_filter[3], nb_filter[2], kernel_size, 1, imgsize[2])

        self.DB1_2 = block(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], kernel_size, 1, imgsize[0])
        self.DB2_2 = block(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], kernel_size, 1, imgsize[1])
        self.DB1_3 = block(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], kernel_size, 1, imgsize[0])

        if self.deepsupervision:
            self.conv1 = ConvLayer(nb_filter[0], output_nc, 1, stride)
            self.conv2 = ConvLayer(nb_filter[0], output_nc, 1, stride)
            self.conv3 = ConvLayer(nb_filter[0], output_nc, 1, stride)
        else:
            self.conv_out = ConvLayer(nb_filter[0], output_nc, 1, stride)



    def encoder(self, input):

        x = self.conv0(input)

        x1_0 = self.DB1_0(x)
        x2_0 = self.DB2_0(self.pool(x1_0))
        x3_0 = self.DB3_0(self.pool(x2_0))
        x4_0 = self.DB4_0(self.pool(x3_0))
        return [x1_0, x2_0, x3_0, x4_0]

    def fusion(self, en1, en2, p_type):
        # attention weight
        fusion_function = fusion_strategy.attention_fusion_weight

        f1_0 = fusion_function(en1[0], en2[0], p_type)
        f2_0 = fusion_function(en1[1], en2[1], p_type)
        f3_0 = fusion_function(en1[2], en2[2], p_type)
        f4_0 = fusion_function(en1[3], en2[3], p_type)
        return [f1_0, f2_0, f3_0, f4_0]

    def decoder_train(self, f_en):

        x1_1 = self.DB1_1(torch.cat([f_en[0], self.up(f_en[1])], 1))

        x2_1 = self.DB2_1(torch.cat([f_en[1], self.up(f_en[2])], 1))

        x1_2 = self.DB1_2(torch.cat([f_en[0], x1_1, self.up(x2_1)], 1))

        x3_1 = self.DB3_1(torch.cat([f_en[2], self.up(f_en[3])], 1))

        x2_2 = self.DB2_2(torch.cat([f_en[1], x2_1, self.up(x3_1)], 1))

        x1_3 = self.DB1_3(torch.cat([f_en[0], x1_1, x1_2, self.up(x2_2)], 1))

        if self.deepsupervision:
            output1 = self.conv1(x1_1)
            output2 = self.conv2(x1_2)
            output3 = self.conv3(x1_3)

            return [output1, output2, output3]
        else:
            output = self.conv_out(x1_3)
            return [output]

    def decoder_eval(self, f_en, tp):


        x1_1 = self.DB1_1(torch.cat([f_en[0], self.up_eval(f_en[0], f_en[1])], 1))

        x2_1 = self.DB2_1(torch.cat([f_en[1], self.up_eval(f_en[1], f_en[2])], 1))

        x1_2 = self.DB1_2(torch.cat([f_en[0], x1_1, self.up_eval(f_en[0], x2_1)], 1))

        x3_1 = self.DB3_1(torch.cat([f_en[2], self.up_eval(f_en[2], f_en[3])], 1))

        x2_2 = self.DB2_2(torch.cat([f_en[1], x2_1, self.up_eval(f_en[1], x3_1)], 1))

        x1_3 = self.DB1_3(torch.cat([f_en[0], x1_1, x1_2, self.up_eval(f_en[0], x2_2)], 1))


        if self.deepsupervision:
            output1 = self.conv1(x1_1)
            output2 = self.conv2(x1_2)
            output3 = self.conv3(x1_3)
            return [output1, output2, output3]
        else:
            output = self.conv_out(x1_3)
            return [output]


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class qkv_transform(nn.Conv1d):
    """Conv1d for qkv_transform"""


class AxialAttention(nn.Module):
    def __init__(self, in_planes, out_planes, groups=8, kernel_size=56, stride=1, bias=False, width=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        super(AxialAttention, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.width = width

        # Multi-head self attention
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1, padding=0,
                                           bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)

        self.bn_output = nn.BatchNorm1d(out_planes * 2)

        # Position embedding
        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, kernel_size * 2 - 1), requires_grad=True)
        query_index = torch.arange(kernel_size).unsqueeze(0)
        key_index = torch.arange(kernel_size).unsqueeze(1)
        relative_index = key_index - query_index + kernel_size - 1
        self.register_buffer('flatten_index', relative_index.view(-1))
        if stride > 1:
            self.pooling = nn.AvgPool2d(stride, stride=stride)

        self.reset_parameters()

    def forward(self, x):

        if self.width:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)  # N, W, C, H
        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)

        qkv = self.bn_qkv(self.qkv_transform(x))

        midx = qkv.reshape(N * W, self.groups, self.group_planes * 2, H)
        q, k, v = torch.split(midx, [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)


        # Calculate position embedding

        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes * 2,
                                                                                       self.kernel_size,
                                                                                       self.kernel_size)


        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings,
                                                            [self.group_planes // 2, self.group_planes // 2,
                                                             self.group_planes], dim=0)

        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)
        qk = torch.einsum('bgci, bgcj->bgij', q, k)

        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W, 3, self.groups, H, H).sum(dim=1)

        similarity = F.softmax(stacked_similarity, dim=3)
        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)
        stacked_output = torch.cat([sv, sve], dim=-1).view(N * W, self.out_planes * 2, H)
        output = self.bn_output(stacked_output).view(N, W, self.out_planes, 2, H).sum(dim=-2)

        if self.width:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        if self.stride > 1:
            output = self.pooling(output)

        return output

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))

        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))


class AxialBlock(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1,
                 norm_layer=None, kernel_size=56):
        super(AxialBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.))

        self.conv_down = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.hight_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size)
        self.width_block = AxialAttention(width, width, groups=groups, kernel_size=kernel_size, stride=stride,
                                          width=True)

        self.conv_up = conv1x1(width, planes * self.expansion)
        self.bn2 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.hight_block(out)
        out = self.width_block(out)
        out = self.relu(out)

        out = self.conv_up(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
