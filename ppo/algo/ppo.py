import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import copy
import math
import os
import shutil
import pandas as pd

class PPO():
    def __init__(self,
                 actor_critic,
                 macro_critic,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 device = None,
                 lr=None,
                 macro_lr = None, 
                 eps=None,
                 max_grad_norm=None,
                 use_clipped_value_loss=True,
                 writer = None):

        self.actor_critic = actor_critic
        self.macro_critic = macro_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.device = device

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
        self.macro_critic_optimizer = optim.Adam(macro_critic.parameters(), lr=macro_lr, eps=eps)
        self.macro_lr = macro_lr

        self.writer = writer

    def update_macro_critic(self, rollouts, model_path):
        with torch.cuda.device("cuda:{}".format(rollouts.args.cuda_id)):
            torch.cuda.empty_cache()

        print('update...')
        
        for epoch in tqdm(range(rollouts.args.macro_critic_epoch)):

            for param_group in self.macro_critic_optimizer.param_groups:
                param_group['lr'] = self.macro_lr - (self.macro_lr - 1e-5) * epoch / rollouts.args.macro_critic_epoch
            
            data_iter = rollouts.feed_forward_value(mini_batch_size = rollouts.args.macro_critic_batch, model_path=model_path, first=(epoch==0))

            macro_loss_all = 0.0
            num = 0.0

            S = []

            for index, sample in enumerate(data_iter):

                return_batch, action_batch, state_batch = sample

                return_batch = return_batch.unsqueeze(dim=1)

                if state_batch.shape[0]>0:

                    S.append(state_batch.shape[1])

                    if rollouts.args.dataset=='beijing':

                        if state_batch.shape[1]>=500 and state_batch.shape[1]<1000:

                            split_batch = int(state_batch.shape[0] * 1/3)
                        
                        if state_batch.shape[1]>=1000 and state_batch.shape[1]<2000:

                            split_batch = int(state_batch.shape[0] * 1/8)

                        if state_batch.shape[1]>=2000 and state_batch.shape[1]<5000:
                            split_batch = int(state_batch.shape[0] * 1/24)

                        elif state_batch.shape[1]>=5000:
                            split_batch = 2

                        else:
                            split_batch = state_batch.shape[0]

                    if rollouts.args.dataset in ['shanghai','Senegal']:

                        if state_batch.shape[1]>=300 and state_batch.shape[1]<500:

                            split_batch = int(state_batch.shape[0] * 1/2)

                        if state_batch.shape[1]>=500 and state_batch.shape[1]<1000:
                            split_batch = int(state_batch.shape[0] * 1/4)

                        elif state_batch.shape[1]>=1000 and state_batch.shape[1]<2000:
                            split_batch = int(state_batch.shape[0] * 1/16)

                        elif state_batch.shape[1]>=2000  and state_batch.shape[1]<4000:
                            split_batch = int(state_batch.shape[0] * 1/64)

                        elif state_batch.shape[1]>=4000:
                            split_batch = 2

                        else:
                            split_batch = state_batch.shape[0]

                    elif rollouts.args.dataset=='shenzhen':

                        if state_batch.shape[1]>=500 and state_batch.shape[1]<1000:

                            split_batch = int(state_batch.shape[0] * 1/2)
                        
                        if state_batch.shape[1]>=1000 and state_batch.shape[1]<2000:

                            split_batch = int(state_batch.shape[0] * 1/5)

                        if state_batch.shape[1]>=2000 and state_batch.shape[1]<5000:
                            split_batch = int(state_batch.shape[0] * 1/16)

                        elif state_batch.shape[1]>=5000:
                            split_batch = 2

                        else:
                            split_batch = state_batch.shape[0]

                    for split in range(int(math.ceil(state_batch.shape[0]/split_batch))):

                        value_macro = self.macro_critic.evaluate((state_batch[split*split_batch:(split+1)*split_batch].to(self.device), action_batch[split*split_batch:(split+1)*split_batch].to(self.device)))

                        assert value_macro.shape == return_batch[split*split_batch:(split+1)*split_batch].shape
                        value_loss = 0.5 * (return_batch[split*split_batch:(split+1)*split_batch].to(self.device)-value_macro).pow(2).mean()
                        self.macro_critic_optimizer.zero_grad()
                        value_loss.backward()
                        param = nn.utils.clip_grad_norm_(self.macro_critic.parameters(), self.max_grad_norm)
                        self.macro_critic_optimizer.step()
                        macro_loss_all += value_loss.item() * value_macro.shape[0]
                        num += value_macro.shape[0]
                        if torch.isnan(value_macro).any():
                            print('macro value nan')
                            print(torch.isnan(return_batch).any(), torch.isnan(value_macro).any())
                            exit()


            with torch.cuda.device("cuda:{}".format(rollouts.args.cuda_id)):
                torch.cuda.empty_cache()

            #mlflow.log_metric('loss_macro',macro_loss_all/num,step=0)
            self.writer.add_scalar(tag='actor-critic/loss_macro',scalar_value=macro_loss_all/num)



        shutil.rmtree(model_path+'state_save')
        os.mkdir(model_path+'state_save')

    def update(self, rollouts, level=0, flag=0, discriminator=None):

        assert level in [0,1]

        if level==0:
            advantages = rollouts.returns_high - rollouts.value_preds_high
        else:
            advantages = rollouts.returns_low - rollouts.value_preds_low

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)

        #mlflow.log_metric('reward_positive', torch.mean((rollouts.rewards_high >= 0).float()).item(),step=0)
        self.writer.add_scalar(tag='actor-critic/reward_positive',scalar_value=torch.mean((rollouts.rewards_high >= 0).float()).item())

        for _ in range(self.ppo_epoch):

            value_loss_epoch = 0
            action_loss_epoch = 0
            dist_entropy_epoch = 0
            num = 0
            num_b = 0

            ratio_all = []

            data_generator = rollouts.feed_forward_generator(advantages, self.num_mini_batch, None, level,drop = True)

            reward_high, reward_low = 0.0, 0.0

            for sample in data_generator:
                actions_high_batch, actions_low_batch, value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ, uncertainty, state1_batch, state2_batch, length, length_origin = sample

                old_action_log_probs_batch, adv_targ, value_preds_batch, return_batch = old_action_log_probs_batch.to(self.device), adv_targ.to(self.device), value_preds_batch.to(self.device), return_batch.to(self.device)

                # Reshape to do in a single forward pass for all steps

                if level==0:
                    values, action_log_probs, dist_entropy = self.actor_critic.policy_high.evaluate_actions(state1_batch.to(self.device), actions_high_batch.to(self.device), uncertainty.to(self.device), length_origin, length)
                else:
                    values, action_log_probs, dist_entropy = self.actor_critic.policy_low.evaluate_actions((state1_batch.to(self.device), state2_batch.to(self.device), actions_high_batch.to(self.device)), actions_low_batch.to(self.device), length_origin, length)

                assert action_log_probs.shape == old_action_log_probs_batch.shape
                assert action_log_probs.shape == adv_targ.shape
                assert return_batch.shape == values.shape

                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)

                ratio_all.append(ratio.detach().cpu())

                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    assert values.shape == value_preds_batch.shape
                    assert values.shape == return_batch.shape
                    value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()

                if flag==1: # only actor optimization
                    (action_loss - dist_entropy * self.entropy_coef).backward()

                elif flag==2: # only critic optimization
                    (value_loss * self.value_loss_coef).backward()

                elif flag==0: # both actor and critic optimization
                    (value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef).backward()

                param_norm = nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                
                self.optimizer.step()

                value_loss_epoch += value_loss.item() * return_batch.shape[0]
                action_loss_epoch += action_loss.item() * return_batch.shape[0]
                dist_entropy_epoch += dist_entropy.item()
                num += return_batch.shape[0]
                num_b += 1

                if torch.isnan(value_loss).any() or torch.isnan(action_loss).any() or torch.isnan(dist_entropy).any():
                    print('ppo loss nana')
                    exit()

            #ratio_all = torch.cat(ratio_all,dim=0)

            if flag==0:
                if level==0:
                    # mlflow.log_metric('loss_ppo_high_value', value_loss_epoch/num,step=0)
                    # mlflow.log_metric('loss_ppo_high_action',action_loss_epoch/num,step=0)
                    # mlflow.log_metric('loss_ppo_high_entropy',dist_entropy_epoch/num_b,step=0)

                    self.writer.add_scalar(tag='actor-critic/loss_ppo_high_value',scalar_value=value_loss_epoch/num)
                    self.writer.add_scalar(tag='actor-critic/loss_ppo_high_action',scalar_value=action_loss_epoch/num)
                    self.writer.add_scalar(tag='actor-critic/loss_ppo_high_entropy',scalar_value=dist_entropy_epoch/num_b)

                    

                elif level==1:
                    # mlflow.log_metric('loss_ppo_low_value', value_loss_epoch/num,step=0)
                    # mlflow.log_metric('loss_ppo_low_action',action_loss_epoch/num,step=0)
                    # mlflow.log_metric('loss_ppo_low_entropy',dist_entropy_epoch/num_b,step=0)
                    self.writer.add_scalar(tag='actor-critic/loss_ppo_low_value',scalar_value=value_loss_epoch/num)
                    self.writer.add_scalar(tag='actor-critic/loss_ppo_low_action',scalar_value=action_loss_epoch/num)
                    self.writer.add_scalar(tag='actor-critic/loss_ppo_low_entropy',scalar_value=dist_entropy_epoch/num_b)

            elif flag==1:
                # mlflow.log_metric('loss_ppo_cooperative_action',action_loss_epoch/num,step=0)
                # mlflow.log_metric('loss_ppo_cooperative_entropy',dist_entropy_epoch/num_b,step=0)
                self.writer.add_scalar(tag='actor-critic/lloss_ppo_cooperative_action',scalar_value=action_loss_epoch/num)
                self.writer.add_scalar(tag='actor-critic/loss_ppo_cooperative_entropy',scalar_value=dist_entropy_epoch/num_b)
                

            elif flag==2:
                #mlflow.log_metric('loss_ppo_high_value', value_loss_epoch/num,step=0)
                self.writer.add_scalar(tag='actor-critic/loss_ppo_high_value',scalar_value=value_loss_epoch/num)

        if not rollouts.args.no_entropy_decay:
            self.entropy_coef *= 0.5

        