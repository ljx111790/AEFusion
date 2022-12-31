import torch
import torch.nn.functional as F

import numpy as np
# from minpy import numpy as np

from skimage.morphology import disk,square
from skimage.filters.rank import entropy
import torchvision
EPSILON = 1e-5
import os


def attention_fusion_weight(tensor1, tensor2, p_type):

    f_channel = channel_fusion(tensor1, tensor2, p_type)
    f_spatial = spatial_fusion(tensor1, tensor2)
    f_en = en_fusion(tensor1, tensor2)

    tensor_f = (f_channel + f_spatial + f_en) / 3

    return tensor_f



def channel_fusion(tensor1, tensor2, p_type):
    # global max pooling
    shape = tensor1.size()
    # calculate channel attention
    global_p1 = channel_attention(tensor1, p_type)
    global_p2 = channel_attention(tensor2, p_type)

    # get weight map
    global_p_w1 = global_p1 / (global_p1 + global_p2 + EPSILON)
    global_p_w2 = global_p2 / (global_p1 + global_p2 + EPSILON)

    global_p_w1 = global_p_w1.repeat(1, 1, shape[2], shape[3])
    global_p_w2 = global_p_w2.repeat(1, 1, shape[2], shape[3])

    tensor_f = global_p_w1 * tensor1 + global_p_w2 * tensor2

    return tensor_f


def spatial_fusion(tensor1, tensor2, spatial_type='mean'):
    shape = tensor1.size()
    # calculate spatial attention
    spatial1 = spatial_attention(tensor1, spatial_type)
    spatial2 = spatial_attention(tensor2, spatial_type)

    # get weight map, soft-max
    spatial_w1 = torch.exp(spatial1) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)
    spatial_w2 = torch.exp(spatial2) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)

    spatial_w1 = spatial_w1.repeat(1, shape[1], 1, 1)
    spatial_w2 = spatial_w2.repeat(1, shape[1], 1, 1)

    tensor_f = spatial_w1 * tensor1 + spatial_w2 * tensor2

    return tensor_f


def en_fusion(input1, input2, spatial_type='mean'):
    tensor1 = input1
    tensor2 = input2
    shape = tensor1.size()

    tensor1 = tensor1.sum(dim=1, keepdim=True)
    tensor2 = tensor2.sum(dim=1, keepdim=True)

    tensor1 = (tensor1 - torch.min(tensor1)) / (torch.max(tensor1) - torch.min(tensor1))
    tensor2 = (tensor2 - torch.min(tensor2)) / (torch.max(tensor2) - torch.min(tensor2))

    tensor1 = torch.squeeze(tensor1).cpu().numpy()
    tensor2 = torch.squeeze(tensor2).cpu().numpy()

    spatial1 = entropy(tensor1, disk(7))
    spatial2 = entropy(tensor2, disk(7))

    spatial1 = spatial1.astype(np.float32)
    spatial2 = spatial2.astype(np.float32)

    spatial1 = torch.from_numpy(spatial1).cuda()
    spatial2 = torch.from_numpy(spatial2).cuda()

    # get weight map, soft-max
    en_w1 = torch.exp(spatial1) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)
    en_w2 = torch.exp(spatial2) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)

    en_w1 = en_w1.repeat(1, shape[1], 1, 1)
    en_w2 = en_w2.repeat(1, shape[1], 1, 1)

    tensor_f = en_w1 * input1 + en_w2 * input2

    return tensor_f



# channel attention
def channel_attention(tensor, pooling_type='avg'):
    # global pooling
    shape = tensor.size()
    pooling_function = F.avg_pool2d

    if pooling_type is 'attention_avg':
        pooling_function = F.avg_pool2d
    elif pooling_type is 'attention_max':
        pooling_function = F.max_pool2d
    elif pooling_type is 'attention_nuclear':
        pooling_function = nuclear_pooling

    global_p = pooling_function(tensor, kernel_size=shape[2:])
    return global_p


# spatial attention
def spatial_attention(tensor, spatial_type='sum'):
    spatial = []
    if spatial_type is 'mean':
        spatial = tensor.mean(dim=1, keepdim=True)
    elif spatial_type is 'sum':
        spatial = tensor.sum(dim=1, keepdim=True)

    return spatial


# pooling function


def nuclear_pooling(tensor, kernel_size=None):
    shape = tensor.size()
    vectors = torch.zeros(1, shape[1], 1, 1).cuda()
    for i in range(shape[1]):
        u, s, v = torch.svd(tensor[0, i, :, :] + EPSILON)
        s_sum = torch.sum(s)
        vectors[0, i, 0, 0] = s_sum
    return vectors




def imshow(img, name, type):
    # global path
    img = torch.squeeze(img, dim=0)
    nrow = img.size(0)

    img = torchvision.utils.make_grid(img, nrow=nrow, normalize=True, padding=2)

    for i in range(nrow):
        path = './plt_png/' + str(name) + str(type)
        imgname = str(i) + '.jpg'
        if not os.path.exists(path):
            os.makedirs(path)

        torchvision.utils.save_image(img[i], os.path.join(path, imgname))