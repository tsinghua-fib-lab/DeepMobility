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

from ppo import algo, utils
from ppo import get_args
from ppo import Policy, Embedding_Layer, Macro_critic
from ppo import RolloutStorage
from DataLoader import data_loader
from Pretrain import Gravity_pretrain, policy_low_pretrain, policy_high_pretrain
from Evaluation import Evaluation

import setproctitle
import mlflow
from utils import *


def main(args, TIME):
    
    print('Experiment setup...')

    model_path = './ModelSave/{}/'.format(TIME)
    
    setup_init(args)
    
    device = torch.device("cuda:{}".format(args.cuda_id) if args.cuda else "cpu")
    #device = torch.device("cpu")

    LocIndex2lonlat, RegionIndex2lonlat, data, data_test, region2loc, loc2region, region_att, loc_att, real_OD, loclonlat, expert_state_action,init_prob = data_loader(args)

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
            max_grad_norm=args.max_grad_norm)


    # rollouts
    rollouts = RolloutStorage(args, args.num_steps, args.simulate_batch_size) # the first point has no state, thus remove

    evaluation = Evaluation(args, LocIndex2lonlat, RegionIndex2lonlat)

    # optimizer
    disc_optimizer = torch.optim.Adam(discriminator.parameters(),lr=1e-4) # 

    min_distance_step_jsd, min_distance_jsd, min_radius_jsd, min_duration_jsd, min_dailyloc_jsd = 1e9, 1e9,1e9,1e9,1e9

    print('Pretrain actor...')
    x = torch.tensor([[i[2] for i in u] for u in data]).long()

    optimizer = optim.Adam(agent.actor_critic.policy_high.parameters(), lr=3e-3, eps = args.eps)
    criterion = nn.NLLLoss(reduction='sum').to(device)

    if args.uncertainty==0.0:
        args.actor_high_pretrain_epoch = 0

    print('high training')

    for epoch in range(args.actor_high_pretrain_epoch):

        agent.actor_critic.train()

        loss = policy_high_pretrain(x, agent.actor_critic.policy_high, args, device, optimizer, criterion)
        mlflow.log_metric('Loss-PolicyHigh-Pretrain', loss,step=epoch)

    x = (torch.tensor([[i[2] for i in u] for u in data]).long(),torch.tensor([[i[1] for i in u] for u in data]).long())
    optimizer = optim.Adam(agent.actor_critic.policy_low.parameters(), lr=args.lr_pretrain, eps = args.eps)

    print('low training')
    for epoch in range(args.actor_low_pretrain_epoch):

        agent.actor_critic.train()

        loss = policy_low_pretrain(expert_state_action, agent.actor_critic.policy_low, args, device, optimizer)
        mlflow.log_metric('Loss-PolicyLow-Pretrain', loss,step=epoch)
                   
    print('Pretrain gravity...')
    optimizer = optim.Adam(agent.actor_critic.policy_high.actor_others.parameters(), lr=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min',patience=1, min_lr = 1e-5)

    OD_norm = (real_OD/(torch.sum(real_OD,dim=-1, keepdim=True)+1e-9)).reshape(-1,args.total_regions, args.total_regions)
    OD = real_OD.reshape(-1,args.total_regions, args.total_regions)

    args.gravity_pretrain_epoch = 64

    for epoch in range(args.gravity_pretrain_epoch):

        sampler = BatchSampler(SubsetRandomSampler(range(args.total_regions)),args.gravity_batch,drop_last=False)


        Gravity_pretrain(agent.actor_critic.policy_high.actor_others, args, optimizer, scheduler, OD_norm, OD, device, sampler)


    print('############################### Training Start! ###############################')
    for itr in range(0,args.num_updates):

        mlflow.log_metric('epoch',itr,step=0)

        print('itr',itr)

        with torch.no_grad():

            state1, state2, state = init_loc(args, init_prob, region2loc, loc2region, device, batch=args.simulate_batch_size)

            state1_save = copy.deepcopy(state1)
            state2_save = copy.deepcopy(state2)
            state_save = copy.deepcopy(state)

            length = [1] * args.simulate_batch_size
            length_origin = length

            if itr<=args.warmup_iter:
                for param_group in agent.optimizer.param_groups:
                    param_group['lr'] = LR_warmup(args.lr, args.warmup_iter, itr)

            else:
                for param_group in agent.optimizer.param_groups:
                    param_group['lr'] = args.lr - (args.lr-5e-6) * (itr-args.warmup_iter)/args.num_updates


            for step in range(args.num_steps):
                
                # Sample actions
                value_high, action_high, action_log_probs_high, uncertainty, value_low, action_low, action_log_probs_low = agent.actor_critic.act(state_save, length_origin, length)


                value_macro = macro_critic.get_macro_value((state1_save,action_high), length_origin, length)

                masks = torch.FloatTensor([1.0 if step<=args.num_steps-1 else 0.0]).expand(action_high.shape[0])

                rollouts.insert(uncertainty.detach().cpu(), state1_save.detach().cpu(), state2_save.detach().cpu(), action_high.detach().cpu(), action_low.detach().cpu(),\
                action_log_probs_high.detach().cpu(), action_log_probs_low.detach().cpu(), value_high.detach().cpu(), value_low.detach().cpu(), \
                value_macro.detach().cpu(), torch.tensor(length_origin).int(), masks, itr)

                # state update
                state1 = torch.cat((state1, action_high.unsqueeze(dim=1)),dim=1)
                state2 = torch.cat((state2, action_low.unsqueeze(dim=1)),dim=1)
                #state = (state1.to(device), state2.to(device))

                length = [step+2] * args.simulate_batch_size

                length_origin = length

                state1_save = copy.deepcopy(state1)
                state2_save = copy.deepcopy(state2)
                state_save = (state1.to(device), state2.to(device))

                length = [state1_save.shape[1]] * args.simulate_batch_size


            mlflow.log_metric('uncertainty_mean',torch.mean(rollouts.uncertainty.float()).item(),step=itr)

            next_value_high = agent.actor_critic.policy_high.get_value(state1_save, length_origin, length)
            next_value_low = agent.actor_critic.policy_low.get_value((state1_save, state2_save, action_high), length_origin, length)
            next_value_high, next_value_low = next_value_high.detach().cpu(), next_value_low.detach().cpu()
            next_value_macro = macro_critic.get_macro_value((state1_save,action_high), length_origin, length).detach().cpu()


        if args.with_evaluate:
            if itr > 0 and itr % args.evaluate_epoch==0:
                print('Evaluate...')
                distance_step_jsd, distance_jsd, radius_jsd, duration_jsd, dailyloc_jsd = evaluate(init_prob, region2loc, loc2region, agent.actor_critic, args, device, args.evaluate_batch, evaluation, data_test,  step = itr,model_path = model_path, is_save=True)
                
                if distance_step_jsd + distance_jsd + radius_jsd + duration_jsd + dailyloc_jsd < min_distance_step_jsd + min_distance_jsd + min_radius_jsd + min_duration_jsd + min_dailyloc_jsd:
                    mlflow.log_metric('min_epoch',itr,step=itr)
            
                min_distance_step_jsd, min_distance_jsd, min_radius_jsd, min_duration_jsd, min_dailyloc_jsd = min(distance_step_jsd, min_distance_step_jsd), min(min_distance_jsd, distance_jsd), min(min_radius_jsd, radius_jsd), min(min_duration_jsd, duration_jsd), min(min_dailyloc_jsd, dailyloc_jsd)
                
                torch.save(agent.actor_critic.state_dict(), model_path+'model_{}.pkl'.format(itr))

        print('Discriminator training...')
        expert_loader = torch.utils.data.DataLoader(dataset=expert_state_action, batch_size = args.gail_batch_size, shuffle=True, drop_last=True)
        gail_epoch = 15 if itr == 0 else args.gail_epoch

        if itr==1:
            for param_group in disc_optimizer.param_groups:
                param_group['lr'] = args.disc_lr
        
        if itr % args.disc_interval ==0:
            for _ in range(gail_epoch):
                acc_high, acc_low = discriminator.update(expert_loader, rollouts, disc_optimizer)
                if acc_high>0.8:
                    for param_group in disc_optimizer.param_groups:
                        param_group['lr'] = args.disc_lr * 0.1
                    break
                if acc_high<0.6:
                    for param_group in disc_optimizer.param_groups:
                        param_group['lr'] = args.disc_lr

        with torch.no_grad():
            for step in range(args.num_steps):
                length_origin = [step+1] * args.simulate_batch_size
                length = [max(1,int((step+1)/args.history_div))] * args.simulate_batch_size if args.history_div!=0 else length_origin
                rollouts.rewards_high[step] = discriminator.predict_reward_high(rollouts.state_high[step].to(device), rollouts.actions_high[step].squeeze(dim=1).to(device), length_origin, length,  rollouts.masks[step])
                rollouts.rewards_low[step] = discriminator.predict_reward_low((rollouts.state_high[step].to(device),rollouts.state_low[step].to(device)), rollouts.actions_low[step].squeeze(dim=1).to(device),  length_origin, length,  rollouts.masks[step])


        if itr % args.macro_interval == 0 and not args.no_macro and args.macro_coef>0:
            print('Macro critic training......')
            rollouts.micro2macro(real_OD, model_path, init_prob, region2loc,loc2region,macro_critic,actor_critic,device,itr)
            agent.update_macro_critic(rollouts, model_path)
            rollouts.after_update(level=1)

        print('Actor-critic training...')
        if args.no_macro and args.macro_coef==0.0:

            rollouts.compute_returns(next_value_high, next_value_low)

            agent.update(rollouts, level=0, discriminator = discriminator)
            agent.update(rollouts, level=1)
            
            rollouts.after_update(level = 0)

        if itr % args.cooperative_interval == 0 and not args.no_macro and args.macro_coef>0:
            rollouts.compute_returns(next_value_high, next_value_low)
            agent.update(rollouts,level=0,flag=2) # critic
            rollouts.compute_returns(next_value_high, next_value_low, next_value_macro)
            agent.update(rollouts, level=0, flag=1) # actor
            agent.update(rollouts, level=1) # flag=0
        
            rollouts.after_update(level = 0)


if __name__ == "__main__":

    args = get_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_id)
    
    remote_server_uri = "{}"  # mlflow url
    mlflow.set_tracking_uri(remote_server_uri)

    mlflow.set_experiment('TRAINING')

    TIME = model_name()

    setproctitle.setproctitle("{}".format(TIME))

    with mlflow.start_run(run_name = args.run_name+'-'+args.dataset):
        params = {'machine':args.machine,'pretrain_name':args.pretrain_name,'dataset':args.dataset}
        
        mlflow.log_params(params)
        main(args,TIME)
