import numpy as np
import scipy.stats
import pandas as pd
from math import radians, cos, sin, sqrt, asin
import math
from bisect import bisect_right
import json
from scipy import stats
import torch
import torch.nn.functional as F
from tqdm import tqdm
import copy
import pickle
import random

from utils import *

def generation(init_prob, region2loc, loc2region, actor_critic, args, device, pretrained_model_path):

    if pretrained_model_path != '':
        actor_critic.load_state_dict(torch.load(pretrained_model_path,map_location=device), strict=True)
        print('pretrained model loaded!\n')

    batch_size = args.simulate_batch_size

    fake_s1, fake_s2 = [], []

    with torch.no_grad():

        for iindex in tqdm(range(int(args.generate_num/ batch_size))):

            state1, state2, state = init_loc(args, init_prob, region2loc, loc2region, device, batch_size)
            state1_save = copy.deepcopy(state1)
            state2_save = copy.deepcopy(state2)
            state_save = copy.deepcopy(state)

            length = [1] * batch_size
            length_origin = length

            for step in range(args.num_steps):

                # Sample actions
                value_high, action_high, action_log_probs_high, uncertainty, value_low, action_low, action_log_probs_low = actor_critic.act(state_save,length_origin, length)

                # state update
                state1 = torch.cat((state1, action_high.unsqueeze(dim=1)),dim=1)
                state2 = torch.cat((state2, action_low.unsqueeze(dim=1)),dim=1)
                state = (state1.to(device), state2.to(device))

                length = [step + 2] * batch_size

                length_origin = length

                if args.history_div in [2,4]:
                    state1_save = copy.deepcopy(state1[:,-max(1,int(state1.shape[1]/args.history_div)):])
                    state2_save = copy.deepcopy(state1[:,-max(1,int(state2.shape[1]/args.history_div)):])
                    state_save = (state1.to(device), state2.to(device))
                    
                else:
                    state1_save = copy.deepcopy(state1)
                    state2_save = copy.deepcopy(state2)
                    state_save = (state1.to(device), state2.to(device))

                length = [state1_save.shape[1]] * batch_size

                with torch.cuda.device("cuda:{}".format(args.cuda_id)):
                    torch.cuda.empty_cache()
                
            fake_s1 += state1.detach().cpu().numpy().tolist()
            fake_s2 += state2.detach().cpu().numpy().tolist()

    gen_data = [[(fake_s2[user_index][index], fake_s1[user_index][index]) for index in range(len(fake_s1[user_index]))] for user_index in range(len(fake_s1))]
    f = open('./gen_data/gen_data_city_{}_num_{}.pkl'.format(args.dataset, args.generate_num),'wb')
    pickle.dump(gen_data,f)
