import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.nn.functional as F
import torch.optim as optim
import mlflow
import torch.nn as nn
import time
from sklearn.metrics import accuracy_score
import math
import numpy as np

def get_cpc(values1, values2):
    return 2.0 * torch.sum(torch.minimum(values1, values2)) / (torch.sum(values1)+torch.sum(values2))

def Gravity_pretrain(model, args, optimizer, scheduler, OD_norm, OD, device, sampler):

    od_real = []
    od_pred = []

    start = time.time()

    loss_all = 0.0

    for indices in sampler:

        inputs = torch.tensor(indices).long().unsqueeze(dim=0).expand(OD_norm.shape[0],len(indices)).reshape(-1,1).to(device)

        t = torch.arange(int(24/args.hour_agg*int((args.num_steps+1)/24))).long().unsqueeze(dim=1).expand(OD_norm.shape[0],len(indices)).reshape(-1,1).to(device)

        if args.gravity_softmax==1:
            out = F.softmax(model(inputs,t),dim=-1)

        elif args.gravity_softmax==0:
            out = F.relu(model(inputs,t)) + 1e-7
            out_sum = torch.sum(out,dim=-1,keepdim=True)
            out /= out_sum

        real = OD_norm[:,indices].reshape(-1,out.shape[-1]).to(device)

        if args.gravity_loss == 'mse':
            LOSS = nn.L1Loss(reduction='sum')
            loss = LOSS(out, real)

        elif args.gravity_loss == 'cross':
            loss = -torch.sum(torch.log(out+1e-9) * real)

        optimizer.zero_grad()
        loss.backward()    
        optimizer.step()

        lr = optimizer.state_dict()['param_groups'][0]['lr']

        mlflow.log_metric('lr_pretrain_gravity',lr)

        #scheduler.step(loss)

        loss_all += loss.item()

        od_real.append(OD[:,indices].reshape(-1,out.shape[-1]))
        od_pred.append(out.detach().cpu()*torch.sum(OD[:,indices],dim=-1,keepdim=True).reshape(-1,1))

    od_real = torch.cat(od_real,dim=0)
    od_pred = torch.cat(od_pred,dim=0)

    od_pred = od_pred[np.where(od_real>0)]
    od_real = od_real[np.where(od_real>0)]

    cpc1 = get_cpc(od_real, od_pred)

    mlflow.log_metric('pretrain_cpc',cpc1.item())

    mlflow.log_metric('pretrain_gravity_loss',loss_all)


def policy_low_pretrain(expert_state_action, actor, args, device, optimizer):

    batch_size = 2048

    expert_loader = torch.utils.data.DataLoader(dataset=expert_state_action, batch_size = batch_size, shuffle=True, drop_last=False)

    loss_all, num = 0.0, 0.0

    real, pred = [], []

    for index, expert_batch in enumerate(expert_loader):

        expert_state_high, expert_state_low, expert_action_high, expert_action_low, expert_length, expert_length_origin = expert_batch

        expert_length = expert_length.numpy().tolist()

        _, action_log_probs, _  = actor.evaluate_actions((expert_state_high.to(device), expert_state_low.to(device), expert_action_high.to(device)), expert_action_low.to(device), expert_length_origin, expert_length)

        loss = -torch.sum(action_log_probs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_all += loss.item()
        num += action_log_probs.shape[0]

    return loss_all / num


def policy_high_pretrain(x, actor, args, device, optimizer, criterion):

    total_loss = 0.0
    num = 0.0

    batch_size = 16

    for batch in range(int(len(x)/batch_size)):
        
        inputs = x[batch * batch_size : (batch+1) * batch_size].to(device)

        length = (torch.arange(inputs.shape[1]).unsqueeze(dim=0).expand(inputs.shape[0],inputs.shape[1]) + 2).to(device) # 注意这里是+2

        pred = actor.forward(inputs, length)[:,:-1]

        pred = pred.reshape(-1,pred.shape[-1])

        target = x[batch * batch_size : (batch+1) * batch_size, 1:].reshape(-1).to(device)

        loss = criterion(pred, target)

        total_loss += loss.item()

        num += pred.shape[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        with torch.cuda.device("cuda:{}".format(args.cuda_id)):
            torch.cuda.empty_cache()
        
    return total_loss / num