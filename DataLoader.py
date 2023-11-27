import torch
import torch.nn as nn
import json
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import time
import numpy as np
import random

def data_loader(args):
    # data load
    init_prob = np.load('data/{}/{}m/init_distribution.npy'.format(args.dataset, args.resolution)).tolist()
    
    with open('data/{}/{}m/LocIndex2lonlat.json'.format(args.dataset,args.resolution),'r') as f:
        LocIndex2lonlat = json.load(f)
    LocIndex2lonlat = {int(k):LocIndex2lonlat[k] for k in LocIndex2lonlat}

    with open('data/{}/{}m/RegionIndex2lonlat.json'.format(args.dataset,args.resolution),'r') as f:
        RegionIndex2lonlat = json.load(f)
    RegionIndex2lonlat = {int(k):RegionIndex2lonlat[k] for k in RegionIndex2lonlat}
    
    dataloader = DataLoader(args)
    data, data_test, region2loc, loc2region, region_att, loc_att = dataloader.data_load()

    real_OD = torch.tensor(np.load('data/{}/{}m/OD_{}.npy'.format(args.dataset,args.resolution,args.hour_agg))).float()
    real_OD = real_OD[:int((args.num_steps+1)/24)]

    loclonlat = torch.tensor([LocIndex2lonlat[k] for k in range(args.total_locations)]+[(0.0,0.0)])

    expert_state_action = dataloader.ExpertDataset(data)


    return LocIndex2lonlat, RegionIndex2lonlat, data, data_test, region2loc, loc2region, region_att, loc_att, real_OD, loclonlat, expert_state_action,init_prob


class DataLoader(object):

    def __init__(self, args):
        self.args = args

    def data_load(self):
        print('data load...')
        with open('data/{}/{}m/data.json'.format(self.args.dataset,self.args.resolution), 'r') as f:
            data_all = json.load(f)

        self.args.training_num = 20000
        print('training_num', self.args.training_num)

        data = data_all[:self.args.training_num]
        data_test = data_all[self.args.training_num:self.args.training_num+self.args.evaluate_batch]

        with open('data/{}/{}m/user_num.json'.format(self.args.dataset,self.args.resolution),'r') as f:
            user_num = json.load(f)
            self.args.num_days = user_num['num_days']
            self.args.user_num = user_num['num']
        
        self.args.num_steps = self.args.num_days*24-1 # -1 to specify a non-null state

        data = [[loc for loc in user if loc[0]<=self.args.num_steps] for user in data]
        data_test = [[loc for loc in user if loc[0]<=self.args.num_steps] for user in data_test]

        print('num_days:{}, num_user:{}'.format(self.args.num_days, self.args.user_num))
        
        loc_att = None

        with open('data/{}/{}m/region2loc.json'.format(self.args.dataset,self.args.resolution),'r') as f:
            region2loc = torch.tensor(json.load(f)).long() # N_region * loc_num (padding value: -1)

        with open('data/{}/{}m/loc2region.json'.format(self.args.dataset, self.args.resolution),'r') as f:
            loc2region = torch.tensor(json.load(f)).long() # N_loc

        region_att = pd.read_csv('data/{}/{}m/region_attribute.csv'.format(self.args.dataset,self.args.resolution))

        region_att['local_ratio'] = region_att['local'] / region_att['population']

        attribute = ['population','local','cate_num','population_norm_100','population_norm_1000','poi_norm','ratio']
        
        region_att = region_att[['RegionIndex','lonlat']+attribute].values.tolist()

        region_att.sort()

        region_att = [torch.tensor([eval(r[1]) for r in region_att]), \
        torch.tensor([[float(r[2]), float(r[3])]+eval(r[4]) for r in region_att]), \
        torch.tensor([[r[-4] if self.args.region_feature=='population_norm_100' else r[-3]] +eval(r[-2])+eval(r[-1]) for r in region_att])]

        assert len(region_att[0]) == len(region2loc)

        self.args.region_attr_num = len(attribute)
        self.args.total_locations = len(loc2region)
        self.args.total_regions = len(region2loc)

        assert len(region_att[0]) == self.args.total_regions

        print('total_regions:{}, total_locations:{}'.format(self.args.total_regions, self.args.total_locations))
        
        region2loc[region2loc==-1] = self.args.total_locations


        return data, data_test, region2loc, loc2region, region_att, loc_att

    def ExpertDataset(self, data):
        t0, t1, a0, a1, length = [], [], [], [], []

        start = time.time()

        print('data process...')

        t0 = [[i[2] for i in user[:step+1]] + [self.args.total_regions] * (self.args.num_steps-step-1) for user in data for step in range(self.args.num_steps)]
        t1 = [[i[1] for i in user[:step+1]] + [self.args.total_locations] * (self.args.num_steps-step-1) for user in data for step in range(self.args.num_steps)]
        a0 = [user[step+1][2] for user in data for step in range(self.args.num_steps)]
        a1 = [user[step+1][1] for user in data for step in range(self.args.num_steps)]
        #length = [len(i) for i in t0] 不对！！！！！
        length = [step+1 for user in data for step in range(self.args.num_steps)]
        length_origin = [step+1 for user in data for step in range(self.args.num_steps)]

        end = time.time()

        t0, t1, a0, a1, length, length_origin = torch.tensor(t0).long(), torch.tensor(t1).long(), torch.tensor(a0).long(), torch.tensor(a1).long(), torch.tensor(length).long(), torch.tensor(length_origin).long()

        state_action = TensorDataset(t0, t1, a0, a1, length, length_origin)

        print('process time: {} min'.format(round((end-start)/60.0,3)))

        return state_action



