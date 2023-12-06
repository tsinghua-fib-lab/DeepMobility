import torch


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

from utils import *
from Evaluation import evaluate, evaluate_macro

def DeepMobility_train(args, agent, discriminator, data, data_test, expert_state_action, rollouts, evaluation, disc_optimizer, init_prob, region2loc, loc2region, device, model_path, real_OD, writer):
    min_distance_step_jsd, min_distance_jsd, min_radius_jsd, min_duration_jsd, min_dailyloc_jsd = 1e9, 1e9,1e9,1e9,1e9
    print('############################### Training Start! ###############################')
    for itr in range(0,args.num_updates):
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


                value_macro = agent.macro_critic.get_macro_value((state1_save,action_high), length_origin, length)

                masks = torch.FloatTensor([1.0 if step<=args.num_steps-1 else 0.0]).expand(action_high.shape[0])

                rollouts.insert(uncertainty.detach().cpu(), state1_save.detach().cpu(), state2_save.detach().cpu(), action_high.detach().cpu(), action_low.detach().cpu(),\
                action_log_probs_high.detach().cpu(), action_log_probs_low.detach().cpu(), value_high.detach().cpu(), value_low.detach().cpu(), \
                value_macro.detach().cpu(), torch.tensor(length_origin).int(), masks, itr)

                # state update
                state1 = torch.cat((state1, action_high.unsqueeze(dim=1)),dim=1)
                state2 = torch.cat((state2, action_low.unsqueeze(dim=1)),dim=1)

                length = [step+2] * args.simulate_batch_size

                length_origin = length

                state1_save = copy.deepcopy(state1)
                state2_save = copy.deepcopy(state2)
                state_save = (state1.to(device), state2.to(device))

                length = [state1_save.shape[1]] * args.simulate_batch_size
                
            writer.add_scalar(tag='Train/uncertainty_means',scalar_value=torch.mean(rollouts.uncertainty.float()).item(), global_step=itr)
            #mlflow.log_metric('uncertainty_mean',torch.mean(rollouts.uncertainty.float()).item(),step=itr)

            next_value_high = agent.actor_critic.policy_high.get_value(state1_save, length_origin, length)
            next_value_low = agent.actor_critic.policy_low.get_value((state1_save, state2_save, action_high), length_origin, length)
            next_value_high, next_value_low = next_value_high.detach().cpu(), next_value_low.detach().cpu()
            next_value_macro = agent.macro_critic.get_macro_value((state1_save,action_high), length_origin, length).detach().cpu()

        if args.with_evaluate:
            if itr % args.evaluate_epoch==0:
                print('Evaluate...')
                distance_step_jsd, distance_jsd, radius_jsd, duration_jsd, dailyloc_jsd = evaluate(init_prob, region2loc, loc2region, agent.actor_critic, args, device, args.evaluate_batch, evaluation, data_test,  step = itr,model_path = model_path, is_save=True, writer=writer)
                min_distance_step_jsd, min_distance_jsd, min_radius_jsd, min_duration_jsd, min_dailyloc_jsd = min(distance_step_jsd, min_distance_step_jsd), min(min_distance_jsd, distance_jsd), min(min_radius_jsd, radius_jsd), min(min_duration_jsd, duration_jsd), min(min_dailyloc_jsd, dailyloc_jsd)
                torch.save(agent.actor_critic.state_dict(), model_path+'model_{}.pkl'.format(itr))
                OD_fake = evaluate_macro(init_prob, region2loc, loc2region, agent.actor_critic, args, device, real_OD, model_path=model_path, step = itr, total_num=300000, is_save=True, writer=writer)
                

        print('Discriminator training...')
        expert_loader = torch.utils.data.DataLoader(dataset=expert_state_action, batch_size = args.gail_batch_size, shuffle=True, drop_last=True)
        gail_epoch = 15 if itr == 0 else args.gail_epoch

        if itr==1:
            for param_group in disc_optimizer.param_groups:
                param_group['lr'] = args.disc_lr
        
        if itr % args.disc_interval ==0:
            for _ in range(gail_epoch):
                acc_high, acc_low = discriminator.update(expert_loader, rollouts, disc_optimizer,writer = writer, epoch = itr)
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


        writer.add_scalar(tag='Train/reward_high',scalar_value=torch.mean(rollouts.rewards_high).item(), global_step=itr)
        writer.add_scalar(tag='Train/reward_low',scalar_value=torch.mean(rollouts.rewards_low).item(), global_step=itr)
        writer.add_scalar(tag='Train/Q_macro',scalar_value=torch.mean(rollouts.value_macro_combine).item(), global_step=itr)

        if itr % args.macro_interval == 0 and not args.no_macro and args.macro_coef>0:
            print('Macro critic training......')
            rollouts.micro2macro(real_OD, model_path, init_prob, region2loc,loc2region,agent.macro_critic,agent.actor_critic,device,itr)
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
