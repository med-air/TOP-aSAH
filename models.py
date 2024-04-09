# Setup
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

import torch.nn.init as init
from torchvision import models

from resnet3d import *

import math
import numpy as np
import copy
import random

class BasicBlock(nn.Module):
    expansion = 1
    
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)
        
        return out

def binary_classification_loss(concat_true, concat_pred):
    t_true = concat_true[:, 1]
    t_pred = concat_pred[:, 2]
    t_pred = (t_pred + 0.001) / 1.002
    #loss = F.binary_cross_entropy(t_pred,t_true)
    loss_cri = nn.BCELoss()
    loss = loss_cri(t_pred, t_true)
    # print(f"T_true: {t_true}")
    # print(f"T_pred: {t_pred}")
    # print(F.binary_cross_entropy(t_true, t_pred))
    return loss

def multi_classification_loss(concat_true, concat_pred, traumatic):
    t_true = concat_true[:, 1]
    t_pred = concat_pred[:, 3:6]
    # traumatic_index = torch.where(traumatic==1)
    
    # t_true = t_true[traumatic_index]
    # t_pred = t_pred[traumatic_index]
    #t_pred = (t_pred + 0.001) / 1.002
    #loss = F.binary_cross_entropy(t_pred,t_true)
    loss_cri = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.0,2.0,5.0])).float().cuda())
    #loss_cri = nn.CrossEntropyLoss()
    loss = loss_cri(t_pred, t_true.long())
    # print(f"T_true: {t_true}")
    # print(f"T_pred: {t_pred}")
    # print(F.binary_cross_entropy(t_true, t_pred))
    return loss

def multi_classification_loss_ours(concat_true, concat_pred, traumatic):
    t_true = concat_true[:, 1]
    t_pred_1 = concat_pred[:, 3:6]
    t_pred_2 = concat_pred[:, 6:9]
    # traumatic_index = torch.where(traumatic==1)
    
    # t_true = t_true[traumatic_index]
    # t_pred = t_pred[traumatic_index]
    #t_pred = (t_pred + 0.001) / 1.002
    #loss = F.binary_cross_entropy(t_pred,t_true)
    loss_cri = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.0,2.0,5.0])).float().cuda())
    #loss_cri = nn.CrossEntropyLoss()
    loss1 = loss_cri(t_pred_1, t_true.long())
    loss2 = loss_cri(t_pred_2, t_true.long())
    # print(f"T_true: {t_true}")
    # print(f"T_pred: {t_pred}")
    # print(F.binary_cross_entropy(t_true, t_pred))
    return loss1 + loss2

def binary_classification_loss_outcome(concat_true, concat_pred):
    y_true = concat_true[:, 0]
    t_true = concat_true[:, 1]

    y0_pred = concat_pred[:, 0]
    y1_pred = concat_pred[:, 1]

    loss0 = torch.sum((1. - t_true) * F.binary_cross_entropy(y0_pred,y_true,reduction = 'none'))
    loss1 = torch.sum(t_true * F.binary_cross_entropy(y1_pred,y_true,reduction = 'none'))

    return loss0 / (torch.sum((1. - t_true))+1e-8) + loss1 / (torch.sum(t_true)+1e-8)

def multi_classification_loss_outcome(concat_true, concat_pred, traumatic, class_ratio):
    traumatic_index = torch.where(traumatic==1)
    concat_true = concat_true[traumatic_index]
    concat_pred = concat_pred[traumatic_index]

    y_true = concat_true[:, 0]
    t_true = concat_true[:, 1]

    y0_pred = concat_pred[:, 0]
    y1_pred = concat_pred[:, 1]
    y2_pred = concat_pred[:, 2]

    map_0 = torch.zeros(t_true.shape)
    map_0[t_true==0] = 1
    map_1 = torch.zeros(t_true.shape)
    map_1[t_true==1] = 1
    map_2 = torch.zeros(t_true.shape)
    map_2[t_true==2] = 1
    
    weight_0 = torch.ones(y_true.shape)
    weight_0[y_true==1] = 2*(1/class_ratio[0])/(1/class_ratio[0]+1/(1 - class_ratio[0]))
    weight_0[y_true==0] = 2*(1/(1 - class_ratio[0]))/(1/class_ratio[0]+1/(1 - class_ratio[0]))
    weight_1 = torch.ones(y_true.shape)
    weight_1[y_true==1] = 2*(1/class_ratio[1])/(1/class_ratio[1]+1/(1 - class_ratio[1]))
    weight_1[y_true==0] = 2*(1/(1 - class_ratio[1]))/(1/class_ratio[1]+1/(1 - class_ratio[1]))
    weight_2 = torch.ones(y_true.shape)
    weight_2[y_true==1] = 2*(1/class_ratio[2])/(1/class_ratio[2]+1/(1 - class_ratio[2]))
    weight_2[y_true==0] = 2*(1/(1 - class_ratio[2]))/(1/class_ratio[2]+1/(1 - class_ratio[2]))

    loss0 = torch.sum(map_0.cuda() * F.binary_cross_entropy(y0_pred,y_true,reduction = 'none'))
    loss1 = torch.sum(map_1.cuda() * F.binary_cross_entropy(y1_pred,y_true,reduction = 'none'))
    loss2 = torch.sum(map_2.cuda() * F.binary_cross_entropy(y2_pred,y_true,reduction = 'none'))

    return loss0 / (torch.sum(map_0)+1e-8) + loss1 / (torch.sum(map_1)+1e-8) + loss2 / (torch.sum(map_2)+1e-8)


def multi_classification_loss_outcome_woweight(concat_true, concat_pred, traumatic, class_ratio):
    traumatic_index = torch.where(traumatic==1)
    concat_true = concat_true[traumatic_index]
    concat_pred = concat_pred[traumatic_index]

    y_true = concat_true[:, 0]
    t_true = concat_true[:, 1]

    y0_pred = concat_pred[:, 0]
    y1_pred = concat_pred[:, 1]
    y2_pred = concat_pred[:, 2]

    map_0 = torch.zeros(t_true.shape)
    map_0[t_true==0] = 1
    map_1 = torch.zeros(t_true.shape)
    map_1[t_true==1] = 1
    map_2 = torch.zeros(t_true.shape)
    map_2[t_true==2] = 1
    
    weight_0 = torch.ones(y_true.shape)
    weight_0[y_true==1] = 2*(1/class_ratio[0])/(1/class_ratio[0]+1/(1 - class_ratio[0]))
    weight_0[y_true==0] = 2*(1/(1 - class_ratio[0]))/(1/class_ratio[0]+1/(1 - class_ratio[0]))
    weight_1 = torch.ones(y_true.shape)
    weight_1[y_true==1] = 2*(1/class_ratio[1])/(1/class_ratio[1]+1/(1 - class_ratio[1]))
    weight_1[y_true==0] = 2*(1/(1 - class_ratio[1]))/(1/class_ratio[1]+1/(1 - class_ratio[1]))
    weight_2 = torch.ones(y_true.shape)
    weight_2[y_true==1] = 2*(1/class_ratio[2])/(1/class_ratio[2]+1/(1 - class_ratio[2]))
    weight_2[y_true==0] = 2*(1/(1 - class_ratio[2]))/(1/class_ratio[2]+1/(1 - class_ratio[2]))

    loss0 = torch.sum(map_0.cuda() * F.binary_cross_entropy(y0_pred,y_true,reduction = 'none'))
    loss1 = torch.sum(map_1.cuda() * F.binary_cross_entropy(y1_pred,y_true,reduction = 'none'))
    loss2 = torch.sum(map_2.cuda() * F.binary_cross_entropy(y2_pred,y_true,reduction = 'none'))

    return loss0 / (torch.sum(map_0)+1e-8) + loss1 / (torch.sum(map_1)+1e-8) + loss2 / (torch.sum(map_2)+1e-8)

def regression_loss(concat_true, concat_pred):
    y_true = concat_true[:, 0]
    t_true = concat_true[:, 1]

    y0_pred = concat_pred[:, 0]
    y1_pred = concat_pred[:, 1]

    loss0 = torch.sum((1. - t_true) * torch.square(y_true - y0_pred))
    loss1 = torch.sum(t_true * torch.square(y_true - y1_pred))

    return loss0 + loss1


def ned_loss(concat_true, concat_pred):
    t_true = concat_true[:, 1]

    t_pred = concat_pred[:, 1]
    return torch.sum(F.binary_cross_entropy_with_logits(t_true, t_pred))


def dead_loss(concat_true, concat_pred):
    return regression_loss(concat_true, concat_pred)


def dragonnet_loss_binarycross(concat_pred, concat_true):
    #return regression_loss(concat_true, concat_pred) + binary_classification_loss(concat_true, concat_pred)
    #return binary_classification_loss(concat_true, concat_pred) + binary_classification_loss_outcome(concat_true, concat_pred)
    return binary_classification_loss(concat_true, concat_pred) + binary_classification_loss_outcome(concat_true, concat_pred)


def dragonnet_loss_binarycross_3cls(concat_pred, concat_true, traumatic, class_ratio):
    #return regression_loss(concat_true, concat_pred) + binary_classification_loss(concat_true, concat_pred)
    #return binary_classification_loss(concat_true, concat_pred) + binary_classification_loss_outcome(concat_true, concat_pred)
    return multi_classification_loss_outcome(concat_true, concat_pred, traumatic, class_ratio) + multi_classification_loss(concat_true, concat_pred, traumatic)
    
def dragonnet_loss_binarycross_3cls_ours(concat_pred, concat_true, traumatic, class_ratio):
    #return regression_loss(concat_true, concat_pred) + binary_classification_loss(concat_true, concat_pred)
    #return binary_classification_loss(concat_true, concat_pred) + binary_classification_loss_outcome(concat_true, concat_pred)
    return multi_classification_loss_ours(concat_true, concat_pred, traumatic) + multi_classification_loss_outcome(concat_true, concat_pred, traumatic, class_ratio)

def dragonnet_loss_binarycross_3cls_ours_woCD(concat_pred, concat_true, traumatic, class_ratio):
    #return regression_loss(concat_true, concat_pred) + binary_classification_loss(concat_true, concat_pred)
    #return binary_classification_loss(concat_true, concat_pred) + binary_classification_loss_outcome(concat_true, concat_pred)
    return multi_classification_loss_outcome(concat_true, concat_pred, traumatic, class_ratio)

class EpsilonLayer(nn.Module):
    def __init__(self):
        super(EpsilonLayer, self).__init__()

        # building epsilon trainable weight
        self.weights = nn.Parameter(torch.Tensor(1, 1))

        # initializing weight parameter with RandomNormal
        nn.init.normal_(self.weights, mean=0, std=0.05)

    def forward(self, inputs):
        return torch.mm(torch.ones_like(inputs)[:, 0:1], self.weights.T)


def make_tarreg_loss(ratio=1., dragonnet_loss=dragonnet_loss_binarycross):
    """
    Create the targeted regularization loss criterion
    Args:
        ratio: Ratio of targeted regularization to use
        dragonnet_loss: Simple loss

    """
    
    
    def tarreg_ATE_unbounded_domain_loss(concat_pred, concat_true):
        vanilla_loss = dragonnet_loss(concat_pred, concat_true)

        y_true = concat_true[:, 0]
        t_true = concat_true[:, 1]

        y0_pred = concat_pred[:, 0]
        y1_pred = concat_pred[:, 1]
        t_pred = concat_pred[:, 2]

        epsilons = concat_pred[:, 3]
        
        print(f"Epsilon: {epsilons}")
        t_pred = (t_pred + 0.01) / 1.02
        # t_pred = tf.clip_by_value(t_pred,0.01, 0.99,name='t_pred')

        y_pred = t_true * y1_pred + (1 - t_true) * y0_pred

        h = t_true / t_pred - (1 - t_true) / (1 - t_pred)

        y_pert = y_pred + epsilons * h
        targeted_regularization = torch.sum(torch.square(y_true - y_pert))

        # final
        loss = vanilla_loss + ratio * targeted_regularization
        print(f"Vanilla Loss: {vanilla_loss}")
        print(f"Tarreg: {targeted_regularization}")
        print(f"Tarreg loss: {loss}")
        return loss

    return tarreg_ATE_unbounded_domain_loss

def make_tarreg_loss_Ours(ratio=1., dragonnet_loss=dragonnet_loss_binarycross):
    """
    Create the targeted regularization loss criterion
    Args:
        ratio: Ratio of targeted regularization to use
        dragonnet_loss: Simple loss

    """
    
    
    def tarreg_ATE_unbounded_domain_loss(concat_pred, concat_true):
        vanilla_loss = dragonnet_loss(concat_pred, concat_true)

        y_true = concat_true[:, 0]
        t_true = concat_true[:, 1]

        y0_pred = concat_pred[:, 0]
        y1_pred = concat_pred[:, 1]
        t_pred = concat_pred[:, 2]

        epsilons = concat_pred[:, 3]
        
        print(f"Epsilon: {epsilons}")
        t_pred = (t_pred + 0.01) / 1.02
        # t_pred = tf.clip_by_value(t_pred,0.01, 0.99,name='t_pred')

        y_pred = t_true * y1_pred + (1 - t_true) * y0_pred

        h = t_true / t_pred - (1 - t_true) / (1 - t_pred)

        y_pert = y_pred + epsilons * h
        targeted_regularization = torch.sum(torch.square(y_true - y_pert))

        # final
        loss = vanilla_loss + ratio * targeted_regularization
        print(f"Vanilla Loss: {vanilla_loss}")
        print(f"Tarreg: {targeted_regularization}")
        print(f"Tarreg loss: {loss}")
        return loss

    return tarreg_ATE_unbounded_domain_loss

# weight initialization function
def weights_init_normal(params):
    if isinstance(params, nn.Linear):
        torch.nn.init.normal_(params.weight, mean=0.0, std=1.0)
        torch.nn.init.zeros_(params.bias)


# weight initialization function
def weights_init_uniform(params):
    if isinstance(params, nn.Linear):
        limit = math.sqrt(6 / (params.weight[1] + params.weight[0]))
        torch.nn.init.uniform_(params.weight, a=-limit, b=limit)
        torch.nn.init.zeros_(params.bias)

def weights_xainit_uniform(params):
    if isinstance(params, nn.Linear):
        torch.nn.init.xavier_uniform_(params.weight)
        params.bias.data.fill_(0.01)

def weights_kminit_uniform(params):
    if isinstance(params, nn.Linear):
        torch.nn.init.kaiming_uniform_(params.weight)
        #params.bias.data.fill_(0.01)
        torch.nn.init.zeros_(params.bias)

def cal_similarity(v1,v2):
    return torch.sum(v1*v2)

def cal_similarity_detach(v1,v2):
    return torch.sum(v1*v2).detach()

def treatment_index(t,traumatic):
    index_0 = torch.where((t==0)&(traumatic==1))
    index_1 = torch.where(t==1)
    index_2 = torch.where(t==2)
    return index_0, index_1, index_2

def del_tensor_ele(arr,index):
    arr1 = arr[0:index]
    arr2 = arr[index+1:]
    return torch.cat((arr1,arr2),dim=0)


def similarity_matrix(phi_im,phi_cli,t, traumatic):
    phi_im_norm = F.normalize(phi_im,p=2,dim=1)
    phi_cli_norm = F.normalize(phi_cli,p=2,dim=1)
    index_0, index_1, index_2 = treatment_index(t[:,1], traumatic)
    index_0_im = copy.copy(index_0)
    index_1_im = copy.copy(index_1)
    index_2_im = copy.copy(index_2)
    index_0_cli = copy.copy(index_0)
    index_1_cli = copy.copy(index_1)
    index_2_cli = copy.copy(index_2)
    similarity_matrix_01_im = torch.zeros((phi_im.shape[0]),(phi_im.shape[0])).cuda()
    similarity_matrix_02_im = torch.zeros(similarity_matrix_01_im.shape).cuda()
    similarity_matrix_12_im = torch.zeros(similarity_matrix_01_im.shape).cuda()
    similarity_matrix_01_cli = torch.zeros((similarity_matrix_01_im.shape[0]),(phi_im.shape[0])).cuda()
    similarity_matrix_02_cli = torch.zeros(similarity_matrix_01_im.shape).cuda()
    similarity_matrix_12_cli = torch.zeros(similarity_matrix_01_im.shape).cuda()
    
    im_distance_01 = 1000
    im_distance_01_index = [0,0]
    im_distance_02 = 1000
    im_distance_02_index = [0,0]
    im_distance_12 = 1000
    im_distance_12_index = [0,0]
    
    cli_distance_01 = 1000
    cli_distance_01_index = [0,0]
    cli_distance_02 = 1000
    cli_distance_02_index = [0,0]
    cli_distance_12 = 1000
    cli_distance_12_index = [0,0]
    
    for index in index_0[0]:
        for index_ in index_1[0]:
            similarity_matrix_01_im[index,index_] = cal_similarity_detach(phi_im_norm[index],phi_im_norm[index_])
            similarity_matrix_01_cli[index,index_] = cal_similarity_detach(phi_cli_norm[index],phi_cli_norm[index_])
            if similarity_matrix_01_im[index,index_]<im_distance_01:
                im_distance_01 = similarity_matrix_01_im[index,index_]
                im_distance_01_index = [index,index_]
            if similarity_matrix_01_cli[index,index_]<cli_distance_01:
                cli_distance_01 = similarity_matrix_01_cli[index,index_]
                cli_distance_01_index = [index,index_]
    
    for index in index_0[0]:
        for index_ in index_2[0]:
            similarity_matrix_02_im[index,index_] = cal_similarity_detach(phi_im_norm[index],phi_im_norm[index_])
            similarity_matrix_02_cli[index,index_] = cal_similarity_detach(phi_cli_norm[index],phi_cli_norm[index_])
            if similarity_matrix_02_im[index,index_]<im_distance_02:
                im_distance_02 = similarity_matrix_02_im[index,index_]
                im_distance_02_index = [index,index_]
            if similarity_matrix_02_cli[index,index_]<cli_distance_02:
                cli_distance_02 = similarity_matrix_02_cli[index,index_]
                cli_distance_02_index = [index,index_]
            
    for index in index_1[0]:
        for index_ in index_2[0]:
            similarity_matrix_12_im[index,index_] = cal_similarity_detach(phi_im_norm[index],phi_im_norm[index_])
            similarity_matrix_12_cli[index,index_] = cal_similarity_detach(phi_cli_norm[index],phi_cli_norm[index_])
            if similarity_matrix_12_im[index,index_]<im_distance_12:
                im_distance_12 = similarity_matrix_12_im[index,index_]
                im_distance_12_index = [index,index_]
            if similarity_matrix_12_cli[index,index_]<cli_distance_12:
                cli_distance_12 = similarity_matrix_12_cli[index,index_]
                cli_distance_12_index = [index,index_]
    
    index_0_im = del_tensor_ele(index_0_im[0],int((index_0_im[0]==im_distance_01_index[0]).nonzero()))
    if im_distance_01_index[0] != im_distance_02_index[0]:
        index_0_im = del_tensor_ele(index_0_im,int((index_0_im==im_distance_02_index[0]).nonzero()))
    index_0_cli = del_tensor_ele(index_0_cli[0],int((index_0_cli[0]==cli_distance_01_index[0]).nonzero()))
    if cli_distance_01_index[0] != cli_distance_02_index[0]:
        index_0_cli = del_tensor_ele(index_0_cli,int((index_0_cli==cli_distance_02_index[0]).nonzero()))
    
    index_1_im = del_tensor_ele(index_1_im[0],int((index_1_im[0]==im_distance_01_index[1]).nonzero()))
    if im_distance_01_index[1] != im_distance_12_index[0]:
        index_1_im = del_tensor_ele(index_1_im,int((index_1_im==im_distance_12_index[0]).nonzero()))
    index_1_cli = del_tensor_ele(index_1_cli[0],int((index_1_cli[0]==cli_distance_01_index[1]).nonzero()))
    if cli_distance_01_index[1] != cli_distance_12_index[0]:
        index_1_cli = del_tensor_ele(index_1_cli,int((index_1_cli==cli_distance_12_index[0]).nonzero()))
    
    index_2_im = del_tensor_ele(index_2_im[0],int((index_2_im[0]==im_distance_12_index[1]).nonzero()))
    if im_distance_12_index[1] != im_distance_02_index[1]:
        index_2_im = del_tensor_ele(index_2_im,int((index_2_im==im_distance_02_index[1]).nonzero()))
    index_2_cli = del_tensor_ele(index_2_cli[0],int((index_2_cli[0]==cli_distance_12_index[1]).nonzero()))
    if cli_distance_12_index[1] != cli_distance_02_index[1]:
        index_2_cli = del_tensor_ele(index_2_cli,int((index_2_cli==cli_distance_02_index[1]).nonzero()))
    
    random_im_b = random.randint(0,index_0_im.shape[0]-1)
    random_cli_b = random.randint(0,index_0_cli.shape[0]-1)
    random_im_e = random.randint(0,index_1_im.shape[0]-1)
    random_cli_e = random.randint(0,index_1_cli.shape[0]-1)
    random_im_h = random.randint(0,index_2_im.shape[0]-1)
    random_cli_h = random.randint(0,index_2_cli.shape[0]-1)
    
    pairs_im = [im_distance_01_index[0],index_0_im[random_im_b],im_distance_02_index[0],im_distance_01_index[1],index_1_im[random_im_e],im_distance_12_index[0],im_distance_12_index[1],index_2_im[random_im_h],im_distance_01_index[1]]
    pairs_cli = [cli_distance_01_index[0],index_0_cli[random_cli_b],cli_distance_02_index[0],cli_distance_01_index[1],index_1_cli[random_cli_e],cli_distance_12_index[0],cli_distance_12_index[1],index_2_cli[random_cli_h],cli_distance_01_index[1]]
    
    phi_sim_im_ab = cal_similarity_detach(phi_im_norm[pairs_im[0]],phi_im_norm[pairs_im[1]])
    phi_sim_im_ac = cal_similarity_detach(phi_im_norm[pairs_im[0]],phi_im_norm[pairs_im[2]])
    phi_sim_im_bc = cal_similarity_detach(phi_im_norm[pairs_im[1]],phi_im_norm[pairs_im[2]])
    
    phi_sim_im_de = cal_similarity_detach(phi_im_norm[pairs_im[3]],phi_im_norm[pairs_im[4]])
    phi_sim_im_df = cal_similarity_detach(phi_im_norm[pairs_im[3]],phi_im_norm[pairs_im[5]])
    phi_sim_im_ef = cal_similarity_detach(phi_im_norm[pairs_im[4]],phi_im_norm[pairs_im[5]])
    
    phi_sim_im_gh = cal_similarity_detach(phi_im_norm[pairs_im[6]],phi_im_norm[pairs_im[7]])
    phi_sim_im_gi = cal_similarity_detach(phi_im_norm[pairs_im[6]],phi_im_norm[pairs_im[8]])
    phi_sim_im_hi = cal_similarity_detach(phi_im_norm[pairs_im[7]],phi_im_norm[pairs_im[8]])
    
    phi_sim_cli_ab = cal_similarity_detach(phi_cli_norm[pairs_cli[0]],phi_cli_norm[pairs_cli[1]])
    phi_sim_cli_ac = cal_similarity_detach(phi_cli_norm[pairs_cli[0]],phi_cli_norm[pairs_cli[2]])
    phi_sim_cli_bc = cal_similarity_detach(phi_cli_norm[pairs_cli[1]],phi_cli_norm[pairs_cli[2]])
    
    phi_sim_cli_de = cal_similarity_detach(phi_cli_norm[pairs_cli[3]],phi_cli_norm[pairs_cli[4]])
    phi_sim_cli_df = cal_similarity_detach(phi_cli_norm[pairs_cli[3]],phi_cli_norm[pairs_cli[5]])
    phi_sim_cli_ef = cal_similarity_detach(phi_cli_norm[pairs_cli[4]],phi_cli_norm[pairs_cli[5]])
    
    phi_sim_cli_gh = cal_similarity_detach(phi_cli_norm[pairs_cli[6]],phi_cli_norm[pairs_cli[7]])
    phi_sim_cli_gi = cal_similarity_detach(phi_cli_norm[pairs_cli[6]],phi_cli_norm[pairs_cli[8]])
    phi_sim_cli_hi = cal_similarity_detach(phi_cli_norm[pairs_cli[7]],phi_cli_norm[pairs_cli[8]])
    
    sim_im_hub = [phi_sim_im_ab, phi_sim_im_ac, phi_sim_im_bc, phi_sim_im_de, phi_sim_im_df, phi_sim_im_ef, phi_sim_im_gh, phi_sim_im_gi, phi_sim_im_hi]
    sim_cli_hub = [phi_sim_cli_ab, phi_sim_cli_ac, phi_sim_cli_bc, phi_sim_cli_de, phi_sim_cli_df, phi_sim_cli_ef, phi_sim_cli_gh, phi_sim_cli_gi, phi_sim_cli_hi]
    
    return pairs_im, pairs_cli, sim_im_hub, sim_cli_hub
        
class MultiRL(nn.Module):
    """
    3-headed dragonnet architecture
    """

    def __init__(self, in_features, out_features=[200, 100, 1]):
        super(MultiRL, self).__init__()
        dropout = False
        # representation layers 3 : block1
        # units in kera = out_features
        self.representation_block = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features[0]),
            nn.BatchNorm1d(out_features[0]),
            nn.ReLU(),
            nn.Linear(in_features=out_features[0], out_features=out_features[0]),
            nn.BatchNorm1d(out_features[0]),
            nn.ReLU(),
            nn.Linear(in_features=out_features[0], out_features=out_features[0]),
            nn.BatchNorm1d(out_features[0]),
            nn.ReLU()
        )

        # -----------Propensity Head
        self.t_predictions_im = nn.Sequential(nn.Linear(in_features=out_features[0], out_features=3),
                                           nn.Softmax()
                                           )
        self.t_predictions_cli = nn.Sequential(nn.Linear(in_features=out_features[0], out_features=3),
                                           nn.Softmax()
                                           )

        # -----------t0 Head
        self.t0_head = nn.Sequential(nn.Linear(in_features=out_features[0], out_features=out_features[1]),
                                     nn.BatchNorm1d(out_features[1]),
                                     nn.ReLU(),
                                     
                                     nn.Linear(in_features=out_features[1], out_features=out_features[2]),
                                     nn.Sigmoid()
                                     )

        # ----------t1 Head
        self.t1_head = nn.Sequential(nn.Linear(in_features=out_features[0], out_features=out_features[1]),
                                     nn.BatchNorm1d(out_features[1]),
                                     nn.ReLU(),
                                     
                                     nn.Linear(in_features=out_features[1], out_features=out_features[2]),
                                     nn.Sigmoid()
                                     )
        
        self.t2_head = nn.Sequential(nn.Linear(in_features=out_features[0], out_features=out_features[1]),
                                     nn.BatchNorm1d(out_features[1]),
                                     nn.ReLU(),
                                     
                                     nn.Linear(in_features=out_features[1], out_features=out_features[2]),
                                     nn.Sigmoid()
                                     )
        
        self.epsilon = EpsilonLayer()
        c=[64,64,128,256,512]
        layers = [3, 4, 6, 3]
        self.inplanes = c[0]
        self.share = torch.nn.Sequential()
        self.share.add_module('conv1', nn.Conv3d(1, c[0],kernel_size=7, stride=2, padding=0, bias=False))
        self.share.add_module('bn1', nn.BatchNorm3d(c[0]))
        self.share.add_module('relu', nn.ReLU(inplace=True))
        self.share.add_module('maxpool',nn.MaxPool3d(kernel_size=3, stride=2, padding=1))
        self.share.add_module('layer1', self._make_layer(BasicBlock, c[1], layers[0]))
        self.share.add_module('layer2', self._make_layer(BasicBlock, c[2], layers[1], stride=2))
        self.share.add_module('layer3', self._make_layer(BasicBlock, c[3], layers[2], stride=2))
        self.share.add_module('layer4', self._make_layer(BasicBlock, c[4], layers[3], stride=2))
        self.share.add_module('avgpool', nn.AvgPool3d([1,7,7])) 
        if dropout is True:
            self.share.add_module('dropout', nn.Dropout(0.5))
        
        self.H_bar_im = nn.Sequential(nn.Linear(200, 200),nn.BatchNorm1d(200),nn.ReLU())
        self.H_bar_cli = nn.Sequential(nn.Linear(200, 200),nn.BatchNorm1d(200),nn.ReLU())
        
        self.pro_cli = nn.Sequential(nn.Linear(200, 200),nn.BatchNorm1d(200),nn.ReLU(),
                                     nn.Linear(200, 200),nn.BatchNorm1d(200),nn.ReLU())
        self.pro_im = nn.Sequential(nn.Linear(200, 200),nn.BatchNorm1d(200),nn.ReLU(),
                                    nn.Linear(200, 200),nn.BatchNorm1d(200),nn.ReLU())
        
        
        self.global_average= nn.AdaptiveAvgPool1d(15)
        self.resenet_head = nn.Sequential(nn.Linear(512, 200),nn.BatchNorm1d(200),nn.ReLU())
        self.dropout = nn.Dropout(p=0.2)
        #self.fc_cat = nn.Sequential(nn.Linear(400, 200),nn.BatchNorm1d(200),nn.ReLU())
        
        self.fc_im_0 = nn.Sequential(nn.Linear(200, 100),nn.BatchNorm1d(100),nn.ReLU())
        self.fc_im_1 = nn.Sequential(nn.Linear(200, 100),nn.BatchNorm1d(100),nn.ReLU())
        self.fc_im_2 = nn.Sequential(nn.Linear(200, 100),nn.BatchNorm1d(100),nn.ReLU())
        self.fc_cli_0 = nn.Sequential(nn.Linear(200, 100),nn.BatchNorm1d(100),nn.ReLU())
        self.fc_cli_1 = nn.Sequential(nn.Linear(200, 100),nn.BatchNorm1d(100),nn.ReLU())
        self.fc_cli_2 = nn.Sequential(nn.Linear(200, 100),nn.BatchNorm1d(100),nn.ReLU())

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv3d(self.inplanes, planes*block.expansion,kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm3d(planes*block.expansion))
        
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
            
        return nn.Sequential(*layers)

    def init_params(self, std=1):
        """
        Initialize layer parameters. Sample weight from Gaussian distribution
        and bias uniform distribution.

        Args:
            std: Standard deviation of Random normal distribution (default: 1)
        """
        # self.representation_block.apply(weights_kminit_uniform)
        # self.t_predictions.apply(weights_kminit_uniform)
        # self.t0_head.apply(weights_kminit_uniform)
        # self.t1_head.apply(weights_kminit_uniform)
        # self.t2_head.apply(weights_kminit_uniform)
        # self.share.apply(weights_kminit_uniform)
        # self.resenet_head.apply(weights_kminit_uniform)
        # self.fc_cat.apply(weights_kminit_uniform)
        # self.H_bar_im.apply(weights_kminit_uniform)
        # self.H_bar_cli.apply(weights_kminit_uniform)
        # self.mutli_concat.apply(weights_kminit_uniform)
        # self.pro_cli.apply(weights_kminit_uniform)
        # self.pro_im.apply(weights_kminit_uniform)
        
        torch.nn.init.xavier_uniform_(self.representation_block)
        torch.nn.init.xavier_uniform_(self.t_predictions_cli)
        torch.nn.init.xavier_uniform_(self.t_predictions_im)
        torch.nn.init.xavier_uniform_(self.t0_head)
        torch.nn.init.xavier_uniform_(self.t1_head)
        torch.nn.init.xavier_uniform_(self.t2_head)
        torch.nn.init.xavier_uniform_(self.resenet_head)
        #torch.nn.init.xavier_uniform_(self.fc_cat)
        torch.nn.init.xavier_uniform_(self.H_bar_im)
        torch.nn.init.xavier_uniform_(self.H_bar_cli)
        #torch.nn.init.xavier_uniform_(self.mutli_concat)
        torch.nn.init.xavier_uniform_(self.pro_cli)
        torch.nn.init.xavier_uniform_(self.pro_im)
        torch.nn.init.xavier_uniform_(self.fc_im_0)
        torch.nn.init.xavier_uniform_(self.fc_im_1)
        torch.nn.init.xavier_uniform_(self.fc_im_2)
        torch.nn.init.xavier_uniform_(self.fc_cli_0)
        torch.nn.init.xavier_uniform_(self.fc_cli_1)
        torch.nn.init.xavier_uniform_(self.fc_cli_2)
        torch.nn.init.xavier_normal_(self.share)

    def forward(self, cli, t_true, image, traumatic, is_test = False, is_tsne = False):
    
        image = self.share.forward(image)
        image = self.dropout(image)
        image = image[:,:,0,0,0]
        phi_im = self.resenet_head(image)
        phi_cli = self.representation_block(cli)
        
        
        
        psi_im = self.H_bar_im(phi_im)
        psi_cli = self.H_bar_cli(phi_cli)
        
        if not is_test:
            psi_im_pro = self.pro_im(psi_im)
            psi_cli_pro = self.pro_cli(psi_cli)
            pairs_im, pairs_cli, sim_im_hub, sim_cli_hub = similarity_matrix(psi_im_pro, psi_cli_pro, t_true, traumatic)
            
            sim_0_1_im = (psi_im_pro[pairs_cli[0]] - psi_im_pro[pairs_cli[3]])**2
            sim_0_2_im = (psi_im_pro[pairs_cli[0]] - psi_im_pro[pairs_cli[8]])**2
            sim_1_2_im = (psi_im_pro[pairs_cli[5]] - psi_im_pro[pairs_cli[6]])**2
            sim_0_1_cli = (psi_cli_pro[pairs_im[0]] - psi_cli_pro[pairs_im[3]])**2
            sim_0_2_cli = (psi_cli_pro[pairs_im[0]] - psi_cli_pro[pairs_im[8]])**2
            sim_1_2_cli = (psi_cli_pro[pairs_im[5]] - psi_cli_pro[pairs_im[6]])**2
            
            close_loss = torch.sum(sim_0_1_im + sim_0_2_im + sim_1_2_im + sim_0_1_cli + sim_0_2_cli + sim_1_2_cli)
            
            psi_im_pro_norm = F.normalize(psi_im_pro,p=2,dim=1)
            psi_cli_pro_norm = F.normalize(psi_cli_pro,p=2,dim=1)
            
            psi_im_ab = cal_similarity(psi_im_pro_norm[pairs_cli[0]],psi_im_pro_norm[pairs_cli[1]])
            psi_im_ac = cal_similarity(psi_im_pro_norm[pairs_cli[0]],psi_im_pro_norm[pairs_cli[2]])
            psi_im_bc = cal_similarity(psi_im_pro_norm[pairs_cli[1]],psi_im_pro_norm[pairs_cli[2]])
            
            psi_im_de = cal_similarity(psi_im_pro_norm[pairs_cli[3]],psi_im_pro_norm[pairs_cli[4]])
            psi_im_df = cal_similarity(psi_im_pro_norm[pairs_cli[3]],psi_im_pro_norm[pairs_cli[5]])
            psi_im_ef = cal_similarity(psi_im_pro_norm[pairs_cli[4]],psi_im_pro_norm[pairs_cli[5]])
            
            psi_im_gh = cal_similarity(psi_im_pro_norm[pairs_cli[6]],psi_im_pro_norm[pairs_cli[7]])
            psi_im_gi = cal_similarity(psi_im_pro_norm[pairs_cli[6]],psi_im_pro_norm[pairs_cli[8]])
            psi_im_hi = cal_similarity(psi_im_pro_norm[pairs_cli[7]],psi_im_pro_norm[pairs_cli[8]])
            
            psi_cli_ab = cal_similarity(psi_cli_pro_norm[pairs_im[0]],psi_cli_pro_norm[pairs_im[1]])
            psi_cli_ac = cal_similarity(psi_cli_pro_norm[pairs_im[0]],psi_cli_pro_norm[pairs_im[2]])
            psi_cli_bc = cal_similarity(psi_cli_pro_norm[pairs_im[1]],psi_cli_pro_norm[pairs_im[2]])
            
            psi_cli_de = cal_similarity(psi_cli_pro_norm[pairs_im[3]],psi_cli_pro_norm[pairs_im[4]])
            psi_cli_df = cal_similarity(psi_cli_pro_norm[pairs_im[3]],psi_cli_pro_norm[pairs_im[5]])
            psi_cli_ef = cal_similarity(psi_cli_pro_norm[pairs_im[4]],psi_cli_pro_norm[pairs_im[5]])
            
            psi_cli_gh = cal_similarity(psi_cli_pro_norm[pairs_im[6]],psi_cli_pro_norm[pairs_im[7]])
            psi_cli_gi = cal_similarity(psi_cli_pro_norm[pairs_im[6]],psi_cli_pro_norm[pairs_im[8]])
            psi_cli_hi = cal_similarity(psi_cli_pro_norm[pairs_im[7]],psi_cli_pro_norm[pairs_im[8]])
            
            sim_loss = (sim_im_hub[0] - psi_cli_ab)**2 + (sim_im_hub[1] - psi_cli_ac)**2 + (sim_im_hub[2] - psi_cli_bc)**2 + \
            (sim_im_hub[3] - psi_cli_de)**2 + (sim_im_hub[4] - psi_cli_df)**2 + (sim_im_hub[5] - psi_cli_ef)**2 + \
            (sim_im_hub[6] - psi_cli_gh)**2 + (sim_im_hub[7] - psi_cli_gi)**2 + (sim_im_hub[8] - psi_cli_hi)**2 + \
            (sim_cli_hub[0] - psi_im_ab)**2 + (sim_cli_hub[1] - psi_im_ac)**2 + (sim_cli_hub[2] - psi_im_bc)**2 + \
            (sim_cli_hub[3] - psi_im_de)**2 + (sim_cli_hub[4] - psi_im_df)**2 + (sim_cli_hub[5] - psi_im_ef)**2 + \
            (sim_cli_hub[6] - psi_im_gh)**2 + (sim_cli_hub[7] - psi_im_gi)**2 + (sim_cli_hub[8] - psi_im_hi)**2
        
        
        
        psi_im_0 = self.fc_im_0(psi_im)
        psi_im_1 = self.fc_im_1(psi_im)
        psi_im_2 = self.fc_im_2(psi_im)
        
        psi_cli_0 = self.fc_cli_0(psi_cli)
        psi_cli_1 = self.fc_cli_1(psi_cli)
        psi_cli_2 = self.fc_cli_2(psi_cli)
        
        #im_cli_combined = torch.cat((psi_im, psi_cli), 1)
        #im_cli_combined = self.mutli_concat(im_cli_combined)
        
        #x = torch.cat((phi_cli, phi_im), 1)
        #x = self.fc_cat(x)
        
        # ------propensity scores
        propensity_head_im = self.t_predictions_im(phi_im)
        propensity_head_cli = self.t_predictions_cli(phi_cli)
        
        # ------t0
        t0_out = torch.cat((psi_im_0, psi_cli_0), 1)
        t0_out = self.t0_head(t0_out)

        # ------t1
        t1_out = torch.cat((psi_im_1, psi_cli_1), 1)
        t1_out = self.t1_head(t1_out)

        # ------t2
        t2_out = torch.cat((psi_im_2, psi_cli_2), 1)
        t2_out = self.t2_head(t2_out)
        if is_tsne:
            return torch.cat((t0_out, t1_out, t2_out, propensity_head_im, propensity_head_cli), 1), psi_im, psi_cli
        else:
            if not is_test:
                return torch.cat((t0_out, t1_out, t2_out, propensity_head_im, propensity_head_cli), 1), sim_loss, close_loss
            else:
                return torch.cat((t0_out, t1_out, t2_out, propensity_head_im, propensity_head_cli), 1)
   
