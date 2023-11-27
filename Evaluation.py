import numpy as np
import scipy.stats
import pandas as pd
from math import radians, cos, sin, sqrt, asin
import math
from bisect import bisect_right
import json
from scipy import stats
import torch
import copy
import mlflow
import pickle
import random
import tqdm
from utils import *

def geodistance(lng1,lat1,lng2,lat2):
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)]) # 经纬度转换成弧度
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    distance=2*asin(sqrt(a))*6371*1000 # 地球平均半径，6371km
    distance=round(distance/1000,3)
    return distance



class KS_test(object):
    def __init__(self, args, LocIndex2lonlat, RegionIndex2lonlat):
        self.args= args

    def KS(p1,p2):
        for user in p1:
            t = []
            for index, i in enumerate(user):
                if index==0 or i!=t[-1]:
                    t.append(i)
            real.append(t)

        for user in p2:
            t = []
            for index, i in enumerate(user):
                if index==0 or i!=t[-1]:
                    t.append(i)
            fake.append(t)

        # distance_step

        f = [geodistance(self.LocIndex2lonlat[i][0],self.LocIndex2lonlat[i][1], self.LocIndex2lonlat[u[index]][0], self.LocIndex2lonlat[u[index]][1]) for u in fake for index, i in enumerate(u[1:])]
        r = [geodistance(self.LocIndex2lonlat[i][0],self.LocIndex2lonlat[i][1], self.LocIndex2lonlat[u[index]][0], self.LocIndex2lonlat[u[index]][1]) for u in real for index, i in enumerate(u[1:])]

        ks_step = stats.ks_2samp(f,r,alternative='two-sided')


        f = [round(len(u)/self.args.num_days) for u in fake]
        r = [round(len(u)/self.args.num_days) for u in real]

        ks_dailyloc = stats.ks_2samp(f,r,alternative='two-sided')

        f = [self.entropy(u) for u in fake]
        r = [self.entropy(u) for u in real]

        ks_entropy = stats.ks_2samp(f,r,alternative='two-sided')

        real = [[self.LocIndex2lonlat[i] for i in u] for u in real]
        fake = [[self.LocIndex2lonlat[i] for i in u] for u in fake]

        def radius_cal(p):
            c = np.mean(p,axis=0)
            r = np.mean([geodistance(i[0],i[1],c[0],c[1]) for i in p])
            return r

        f = [round(radius_cal(np.array(u))) for u in fake]
        r = [round(radius_cal(np.array(u))) for u in real]

        ks_radius = stats.ks_2samp(f,r,alternative='two-sided')


        real = []
        fake = []

        for user in p1:
            t = []
            for index, i in enumerate(user):
                if index==0 or i!=t[-1][1]:
                    t.append((index,i))
            real += [(i[0]-t[ii][0]) for ii , i in enumerate(t[1:])]

        for user in p2:
            t = []
            for index, i in enumerate(user):
                if index==0 or i!=t[-1][1]:
                    t.append((index,i))
            fake += [(i[0]-t[ii][0]) for ii , i in enumerate(t[1:])]

        ks_duration = stats.ks_2samp(fake,real,alternative='two-sided')

        return  ks_step, ks_dailyloc,ks_entropy, ks_radius, ks_duration

class Evaluation(object):

    def __init__(self, args, LocIndex2lonlat, RegionIndex2lonlat):
        self.args= args

        self.LocIndex2lonlat = LocIndex2lonlat
        self.RegionIndex2lonlat = RegionIndex2lonlat

        if args.dataset == 'beijing':
            self.min1,self.max1,self.bin1,self.min2,self.max2,self.bin2,self.min3, self.max3, self.bin3, self.min4, self.max4, self.bin4 = 0,10,5, 0,250,10, 0,8,4, 0,10,5
        elif args.dataset=='shenzhen':
            self.min1,self.max1,self.bin1,self.min2,self.max2,self.bin2,self.min3, self.max3, self.bin3, self.min4, self.max4, self.bin4 = 0,8,4, 0,70,10, 0,8,4, 0,10,5

        elif args.dataset=='shanghai':
            self.min1,self.max1,self.bin1,self.min2,self.max2,self.bin2,self.min3, self.max3, self.bin3, self.min4, self.max4, self.bin4 = 0,10,5, 0,200,10, 0,8,4, 0,10,5

        elif args.dataset=='Senegal':
            self.min1,self.max1,self.bin1,self.min2,self.max2,self.bin2,self.min3, self.max3, self.bin3, self.min4, self.max4, self.bin4 = 0,10,5, 0,200,10, 0,8,4, 0,15,5


    def arr_to_distribution(self,arr, Min, Max, bins, over=None):
        """
        convert an array to a probability distribution
        :param arr: np.array, input array
        :param min: float, minimum of converted value
        :param max: float, maximum of converted value
        :param bins: int, number of bins between min and max
        :return: np.array, output distribution array
        """

        distribution, base = np.histogram(arr[arr<=Max],bins = bins,range=(Min,Max))
        m = np.array([len(arr[arr>Max])],dtype='int64')
        distribution = np.hstack((distribution,m))


        return distribution, base[:-1]

    def get_js_divergence(self, p1, p2):
        """
        calculate the Jensen-Shanon Divergence of two probability distributions
        :param p1:
        :param p2:
        :return:
        """
        # normalize
        p1 = p1 / (p1.sum()+1e-9)
        p2 = p2 / (p2.sum()+1e-9)
        m = (p1 + p2) / 2
        js = 0.5 * scipy.stats.entropy(p1, m) + 0.5 * scipy.stats.entropy(p2, m)
        return js

    def hour(self,p1,p2):
        
        f = [i[0]%24 for u in p1 for i in u]
        r = [i[0]%24 for u in p2 for i in u]

        f = pd.value_counts(f,normalize=True)
        r = pd.value_counts(r,normalize=True)

        f_list = [(f.keys()[i],f.values[i]) for i in len(f)]
        r_list = [(r.keys()[i],r.values[i]) for i in len(r)]

        for i in range(24):
            if i not in f.keys():
                f_list.append((0,0.0))
            if i not in r.keys():
                r_list.append((0,0.0))

        f_list.sort()
        r_list.sort()

        JSD = self.get_js_divergence(r_list, f_list)

        return JSD

    def week(self,p1,p2):
        if self.args.data=='Tencent':
            f = [[1.0 if i[0]%7>=5 else 0.0 for i in u] for u in p1]
            r = [[1.0 if i[0]%7>=5 else 0.0 for i in u] for u in p2]
            f = [sum(u)/(len(u)+1e-9) for u in f]
            r = [sum(u)/(len(u)+1e-9) for u in r]
        
        MIN = 0
        MAX = 1.0

        bins = 10

        r_list, _ = self.arr_to_distribution(np.array(r), MIN, MAX, bins)
        f_list, _ = self.arr_to_distribution(np.array(f), MIN, MAX, bins)

        JSD = self.get_js_divergence(r_list, f_list)

        return JSD

    def I_rank(u):
        t = pd.value_counts(u,normalize=True)


    def entropy(self,u):
        t = pd.value_counts(u,normalize=True).values
        return -sum(np.log(t)*t)


    def distance_jsd(self,p1,p2, loc_type = 'location'):
        
        real = []
        fake = []

        for user in p1:
            t = []
            for index, i in enumerate(user):
                if index==0 or i!=t[-1]:
                    t.append(i)
            real.append(t)

        for user in p2:
            t = []
            for index, i in enumerate(user):
                if index==0 or i!=t[-1]:
                    t.append(i)
            fake.append(t)

        # distance_step

        f = [geodistance(self.LocIndex2lonlat[i][0],self.LocIndex2lonlat[i][1], self.LocIndex2lonlat[u[index]][0], self.LocIndex2lonlat[u[index]][1]) for u in fake for index, i in enumerate(u[1:])]
        r = [geodistance(self.LocIndex2lonlat[i][0],self.LocIndex2lonlat[i][1], self.LocIndex2lonlat[u[index]][0], self.LocIndex2lonlat[u[index]][1]) for u in real for index, i in enumerate(u[1:])]


        MIN = self.min1
        MAX = self.max1
        bins = self.bin1

        r_list, _ = self.arr_to_distribution(np.array(r), MIN, MAX, bins)
        f_list, _ = self.arr_to_distribution(np.array(f), MIN, MAX, bins)

        JSD_step = self.get_js_divergence(r_list, f_list)


        # distance
        f = [sum([geodistance(self.LocIndex2lonlat[i][0],self.LocIndex2lonlat[i][1], self.LocIndex2lonlat[u[index]][0], self.LocIndex2lonlat[u[index]][1]) for index, i in enumerate(u[1:])]) for u in fake]
        r = [sum([geodistance(self.LocIndex2lonlat[i][0],self.LocIndex2lonlat[i][1], self.LocIndex2lonlat[u[index]][0], self.LocIndex2lonlat[u[index]][1]) for index, i in enumerate(u[1:])]) for u in real]


        MIN = self.min2
        MAX = self.max2
        bins = self.bin2

        r_list, _ = self.arr_to_distribution(np.array(r), MIN, MAX, bins)
        f_list, _ = self.arr_to_distribution(np.array(f), MIN, MAX, bins)


        JSD_distance = self.get_js_divergence(r_list, f_list)

        # daily loc
        f = [round(len(u)/self.args.num_days) for u in fake]
        r = [round(len(u)/self.args.num_days) for u in real]

        MIN = self.min3
        MAX = self.max3
        bins = self.bin3

        r_list, _ = self.arr_to_distribution(np.array(r), MIN, MAX, bins)
        f_list, _ = self.arr_to_distribution(np.array(f), MIN, MAX, bins)

        JSD_dailyloc = self.get_js_divergence(r_list, f_list)


        # entropy
        f = [self.entropy(u) for u in fake]
        r = [self.entropy(u) for u in real]


        # radius
        if loc_type == 'location':
            real = [[self.LocIndex2lonlat[i] for i in u] for u in real]
            fake = [[self.LocIndex2lonlat[i] for i in u] for u in fake]

        elif loc_type == 'region':
            real = [[self.RegionIndex2lonlat[i] for i in u] for u in real]
            fake = [[self.RegionIndex2lonlat[i] for i in u] for u in fake]


        def radius_cal(p):
            c = np.mean(p,axis=0)
            r = np.mean([geodistance(i[0],i[1],c[0],c[1]) for i in p])
            return r

        f = [round(radius_cal(np.array(u))) for u in fake]
        r = [round(radius_cal(np.array(u))) for u in real]

        MIN = self.min4
        MAX = self.max4
        bins = self.bin4

        r_list, _ = self.arr_to_distribution(np.array(r), MIN, MAX, bins)
        f_list, _ = self.arr_to_distribution(np.array(f), MIN, MAX, bins)

        JSD_radius = self.get_js_divergence(r_list, f_list)

        return JSD_step, JSD_distance, JSD_radius, JSD_dailyloc
        

    def duration_jsd(self,p1,p2,loc_type = 'location'):

        real = []
        fake = []

        for user in p1:
            t = []
            for index, i in enumerate(user):
                if index==0 or i!=t[-1][1]:
                    t.append((index,i))
            real += [(i[0]-t[ii][0]) for ii , i in enumerate(t[1:])]

        for user in p2:
            t = []
            for index, i in enumerate(user):
                if index==0 or i!=t[-1][1]:
                    t.append((index,i))
            fake += [(i[0]-t[ii][0]) for ii , i in enumerate(t[1:])]

        MIN = 0
        MAX = 24
        bins = 6

        r_list, _ = self.arr_to_distribution(np.array(real), MIN, MAX, bins)
        f_list, _ = self.arr_to_distribution(np.array(fake), MIN, MAX, bins)

        JSD = self.get_js_divergence(r_list, f_list)

        return JSD



    def get_JSD(self,real,fake, loc_type):
        distance_step_jsd, distance_jsd, radius_jsd, dailyloc_jsd = self.distance_jsd(real,fake,loc_type)

        duration_jsd = self.duration_jsd(real,fake,loc_type)

        return distance_step_jsd, distance_jsd, radius_jsd, duration_jsd, dailyloc_jsd
    

def macro_metric(OD_fake,OD_real,step = 0):
    '''input
    y_pred: N * N
    y_true: N * N
    '''

    similarity = F.cosine_similarity(OD_fake.reshape(-1),OD_real.reshape(-1),dim=-1).double().item()

    mae = torch.mean(torch.abs(OD_fake - OD_real)).item()

    rmse = torch.sqrt(torch.sum((OD_fake - OD_real).pow(2))/OD_real.numel()).item()

    cpc = CPC(OD_fake.reshape(-1).numpy(),OD_real.reshape(-1).numpy())

    smape = SMAPE(OD_fake.reshape(-1).numpy(),OD_real.reshape(-1).numpy())

    nrmse = rmse/np.mean(OD_real.reshape(-1).numpy())

    value = pd.DataFrame([i for i in OD_real.reshape(-1).numpy().tolist()],columns = ['flow'])

    value = value['flow'].describe(percentiles = [0.2,0.4, 0.6, 0.8])

    percen = [0, value['20%'], value['40%'], value['60%'], value['80%'], value['max']]

    mlflow.log_metrics({'Macro-{}-similarity'.format(adj):similarity,'Macro-{}-nrmse'.format(adj):nrmse,'Macro-{}-mae'.format(adj):mae,'Macro-{}-cpc'.format(adj):cpc},step=step)


    for i in range(5):

        real = OD_real[torch.where((OD_real<=percen[i+1]) & (OD_real>percen[i]))]
        fake = OD_fake[torch.where((OD_real<=percen[i+1]) & (OD_real>percen[i]))]

        similarity = F.cosine_similarity(fake.reshape(-1),real.reshape(-1),dim=-1).double().item()

        mae = torch.mean(torch.abs(fake - real)).item()

        cpc = CPC(fake.reshape(-1).numpy(),real.reshape(-1).numpy())


        mlflow.log_metrics({'Macro-{}-cpc-{}'.format(adj,i+1):cpc},step=step)



def evaluate(init_prob, region2loc, loc2region, actor_critic, args, device, batch, evaluation, real_data_all, step, model_path = None, is_save=True):

    step_mlflow = step

    batch_size = 1000 if batch >= 1000 else batch

    fake_s1, fake_s2 = [], []

    actor_critic.eval()

    with torch.no_grad():

        for iindex in tqdm(range(int(batch / batch_size))):

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
    f = open(model_path+'gen_data_{}.pkl'.format(step_mlflow),'wb')
    pickle.dump(gen_data,f)


    real_data = random.sample(real_data_all, batch)

    real_s1 = [[i[2] for i in user] for user in real_data]
    real_s2 = [[i[1] for i in user] for user in real_data]

    if args.eval_dist == 'jsd':
        distance_step_jsd, distance_jsd, radius_jsd, duration_jsd, dailyloc_jsd = evaluation.get_JSD(real_s2,fake_s2, loc_type='location')
        mlflow.log_metrics({'jsd_location_distacne_step':distance_step_jsd, 'jsd_location_distance':distance_jsd,'jsd_location_radius':radius_jsd, 'jsd_location_duration':duration_jsd, 'jsd_location_dailyloc':dailyloc_jsd},step=step_mlflow)

    elif args.eval_dist == 'ks':
        ks_step, ks_dailyloc,ks_entropy, ks_radius, ks_duration = evaluation.KS_test(real_s2,fake_s2)
        mlflow.log_metrics({'ks_step':ks_step,'ks_dailyloc':ks_dailyloc,'ks_entropy':ks_entropy,'ks_radius':ks_radius,'ks_duration':ks_duration})

    actor_critic.train()

    return distance_step_jsd, distance_jsd, radius_jsd, duration_jsd, dailyloc_jsd

def evaluate_macro(init_prob, region2loc, loc2region, actor_critic, args, device, OD_real_origin, model_path,  step, total_num=300000):

    step_mlflow = step

    OD_fake = torch.zeros(OD_real_origin.shape) 

    gen_state_1 = []
    gen_state_2 = []

    actor_critic.eval()

    batch_size = 3

    with torch.no_grad():

        for _ in tqdm(range(int(total_num / batch_size))):

            state1, state2, state = init_loc(args, init_prob, region2loc, loc2region, device, batch_size)

            length = [1] * args.simulate_batch_size
            length_origin = length

            for step in range(args.num_steps):
                
                # Sample actions
                value_high, action_high, action_log_probs_high, uncertainty, value_low, action_low, action_log_probs_low = actor_critic.act(state,length_origin, length)

                # state update
                state1 = torch.cat((state1, action_high.unsqueeze(dim=1)),dim=1)
                state2 = torch.cat((state2, action_low.unsqueeze(dim=1)),dim=1)
                state = (state1.to(device), state2.to(device))

                length = [step + 2] * batch_size

            state1 = state1.detach().cpu().numpy().tolist()
            state2 = state2.detach().cpu().numpy().tolist()

            gen_state_1 += state1
            gen_state_2 += state2

            for user_index in range(len(state1)):
                t1 = []
                t2 = []
                for index, i in enumerate(state1[user_index]):
                    if index==0 or state2[user_index][index]!=t2[-1]:
                        if index>0:
                            day, hour = int(index // 24), int((index % 24) // args.hour_agg)
                            OD_fake[day][hour][t1[-1]][i] += 1
                        t1.append(i)
                        t2.append(state2[user_index][index])

            with torch.cuda.device("cuda:{}".format(args.cuda_id)):
                torch.cuda.empty_cache()

    np.save(model_path+'OD_fake_{}.npy'.format(step_mlflow),OD_fake)

    OD_real = OD_real_origin * total_num/args.user_num

    OD_fake_total = copy.deepcopy(OD_fake)
    OD_real_total = copy.deepcopy(OD_real)

    for i in range(OD_fake_total.shape[0]):
        OD_fake_total[i][i] = 0
        OD_real_total[i][i] = 0

    OD_fake_total = OD_fake_total[torch.where(OD_real_total>0)]
    OD_real_total = OD_real_total[torch.where(OD_real_total>0)]

    macro_metric(OD_fake_total, OD_real_total,step = step_mlflow)

    actor_critic.train()

    return OD_fake