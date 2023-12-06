import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.nn.functional as F
import torch.optim as optim

import torch.nn as nn
import time
from sklearn.metrics import accuracy_score
import math
import numpy as np
from utils import *
import torch



def Gravity_pretrain(epoch, model, args, optimizer, scheduler, OD_norm, OD, device, sampler, writer):

    od_real = []
    od_pred = []

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

        loss_all += loss.item()

        od_real.append(OD[:,indices].reshape(-1,out.shape[-1]))
        od_pred.append(out.detach().cpu()*torch.sum(OD[:,indices],dim=-1,keepdim=True).reshape(-1,1))

    od_real = torch.cat(od_real,dim=0)
    od_pred = torch.cat(od_pred,dim=0)

    od_pred = od_pred[np.where(od_real>0)]
    od_real = od_real[np.where(od_real>0)]

    cpc1 = get_cpc(od_real, od_pred)

    writer.add_scalar('Pretrain/cpc',cpc1.item(),global_step = epoch)

    writer.add_scalar('Pretrain/pretrain_gravity_loss',loss_all,global_step = epoch)


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

        length = (torch.arange(inputs.shape[1]).unsqueeze(dim=0).expand(inputs.shape[0],inputs.shape[1]) + 2).to(device) 

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

def pretrain_together(args,data,expert_state_action,agent, real_OD,device, writer):
    print('Pretrain actor...')
    x = torch.tensor([[i[2] for i in u] for u in data]).long()

    optimizer = optim.Adam(agent.actor_critic.policy_high.parameters(), lr=3e-3, eps = args.eps)
    criterion = nn.NLLLoss(reduction='sum').to(device)

    print('high training')

    for epoch in range(args.actor_high_pretrain_epoch):
        agent.actor_critic.train()
        Lr_reduce(optimizer.param_groups, epoch, args.actor_high_pretrain_epoch)
        loss = policy_high_pretrain(x, agent.actor_critic.policy_high, args, device, optimizer, criterion)
        writer.add_scalar(tag='Pretrain/High_loss',scalar_value=loss, global_step=epoch) 

    x = (torch.tensor([[i[2] for i in u] for u in data]).long(),torch.tensor([[i[1] for i in u] for u in data]).long())

    optimizer = optim.Adam(agent.actor_critic.policy_low.parameters(), lr=args.lr_pretrain, eps = args.eps)

    print('low training')
    for epoch in range(args.actor_low_pretrain_epoch):

        agent.actor_critic.train()
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-4   

        loss = policy_low_pretrain(expert_state_action, agent.actor_critic.policy_low, args, device, optimizer)
        writer.add_scalar(tag='Pretrain/Low_loss',scalar_value=loss, global_step=epoch) 


    if args.uncertainty < 1e10:
        print('Pretrain gravity...')
        optimizer = optim.Adam(agent.actor_critic.policy_high.actor_others.parameters(), lr=1e-2)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min',patience=1, min_lr = 1e-5)

        OD_norm = (real_OD/(torch.sum(real_OD,dim=-1, keepdim=True)+1e-9)).reshape(-1,args.total_regions, args.total_regions)
        OD = real_OD.reshape(-1,args.total_regions, args.total_regions)

        args.gravity_pretrain_epoch = 64

        for epoch in range(args.gravity_pretrain_epoch):

            sampler = BatchSampler(SubsetRandomSampler(range(args.total_regions)),args.gravity_batch,drop_last=False)

            if epoch<=5:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = LR_warmup(1e-2, 5, epoch)
            
            elif epoch <= 32:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 1e-2 - (epoch-5) * ((1e-2)-(1e-3))/(100-5)

            elif epoch <= 64:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = 1e-3 - (epoch-100) * ((1e-3)-(1e-4))/100

            Gravity_pretrain(epoch, agent.actor_critic.policy_high.actor_others, args, optimizer, scheduler, OD_norm, OD, device, sampler, writer)
