import copy
import glob
import os
import time
from collections import deque
from tqdm import tqdm
import pandas as pd
import random
import json
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from ppo import algo, utils
from ppo import get_args
from ppo import Policy, Embedding_Layer, Macro_critic
from ppo import RolloutStorage
from DataLoader import DataLoader
from Pretrain import pretrain_together
from Evaluation import Evaluation

import setproctitle
from utils import *
from train import DeepMobility_train


def main(args, TIME):
    
    print('Experiment setup...')

    model_path = './ModelSave/{}/'.format(TIME)

    if not os.path.exists('./ModelSave/'):
        os.mkdir('./ModelSave/')

    if not os.path.exists(model_path):
        os.mkdir(model_path)
        os.mkdir(model_path+'state_save/')
        os.mkdir(model_path+'OD_matrix/')
    
    setup_init(args)
    
    writer = SummaryWriter(log_dir = './tensorboard_log/TIME/',flush_secs=5)

    device = torch.device("cuda:{}".format(args.cuda_id) if args.cuda else "cpu")
    #device = torch.device("cpu")

    
    dataloader = DataLoader(args)
    data, data_test, region2loc, loc2region, region_att, loc_att, init_prob, loclonlat, LocIndex2lonlat, RegionIndex2lonlat, real_OD = dataloader.data_load()
    
    expert_state_action = dataloader.ExpertDataset(data)

    print('Model setup...')
    embedding_layer = Embedding_Layer(args).to(device)
    macro_critic = Macro_critic(args, embedding_layer, device).to(device)
    actor_critic = Policy(args,embedding_layer, region2loc, region_att, loclonlat, device, history_len=args.history_len).to(device)
    discriminator = algo.Discriminator(args, embedding_layer,device).to(device)

    agent = algo.PPO(
            actor_critic,
            macro_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            device = device,
            lr=args.lr,
            macro_lr = args.macro_lr, 
            eps=args.eps,
            max_grad_norm=args.max_grad_norm,
            writer = writer)


    # rollouts
    rollouts = RolloutStorage(args, args.num_steps, args.simulate_batch_size) # the first point has no state, thus remove
    evaluation = Evaluation(args, LocIndex2lonlat, RegionIndex2lonlat)

    # optimizer
    disc_optimizer = torch.optim.Adam(discriminator.parameters(),lr=1e-4) 

    pretrain_together(args, data, expert_state_action, agent, real_OD,device, writer)

    DeepMobility_train(args, agent, discriminator, data, data_test, expert_state_action, rollouts, evaluation, disc_optimizer, init_prob, region2loc, loc2region, device, model_path, real_OD, writer)
    


if __name__ == "__main__":

    args = get_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_id)

    TIME = model_name()
   
    main(args,TIME)