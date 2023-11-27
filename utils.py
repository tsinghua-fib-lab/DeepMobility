
import numpy as np
import scipy.stats
import pandas as pd
from math import radians, cos, sin, sqrt, asin
import math
from bisect import bisect_right
import json

def LR_warmup(lr,epoch_num, epoch_current):
    return lr * epoch_current / epoch_num

def MAPE(Y_Predicted,Y_actual):
    assert (Y_actual > 0).all()
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))
    return mape

def SMAPE(a, f):
    return 1/len(a) * np.sum(2 * np.abs(f-a) / (np.abs(a) + np.abs(f)))

def CPC(y_true, y_pred):
    return 2*np.sum(np.minimum(y_true,y_pred))/np.sum(y_true+y_pred)


def geodistance(lng1,lat1,lng2,lat2):
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)]) # 经纬度转换成弧度
    dlon=lng2-lng1
    dlat=lat2-lat1
    a=sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    distance=2*asin(sqrt(a))*6371*1000 # 地球平均半径，6371km
    distance=round(distance/1000,3)
    return distance

def NRMSE(pred,test):
    return np.sum((pred-test)**2)/np.sum(test**2)


def setup_init(args):
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def init_loc(args, init_prob, region2loc, loc2region, device, batch):
    select_region = np.random.choice(np.arange(args.total_regions), size = batch, p = init_prob)
    region_cnt = pd.value_counts(select_region)
    action_low = [torch.tensor(np.random.choice([i for i in region2loc[k].cpu().numpy().tolist() if i!=args.total_locations], size=(region_cnt[k], 1))) for k in region_cnt.keys()]
    action_low = torch.cat(action_low, dim=0).long().to(device)
    action_high = loc2region[action_low].long().to(device)
    state1 = action_high
    state2 = action_low
    state = (action_high.to(device), action_low.to(device))
    return state1, state2, state


def model_name():
    TIME = int(time.time())
    TIME = time.localtime(TIME)
    return time.strftime("%Y-%m-%d %H:%M:%S",TIME)
