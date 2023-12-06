import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.nn.functional as F
from tqdm import tqdm
import math
import pickle
import numpy as np
import pandas as pd

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


class RolloutStorage(object):

    def __init__(self, args, num_steps, num_processes):

        self.num_processes = num_processes
        self.num_steps = num_steps

        self.gen_batch = min(2500, args.macro_num)

        self.args = args
        self.rewards_high = torch.zeros(num_steps, num_processes, 1)
        self.rewards_low = torch.zeros(num_steps, num_processes, 1)
        self.value_preds_high = torch.zeros(num_steps + 1, num_processes, 1)
        self.value_preds_low = torch.zeros(num_steps + 1, num_processes, 1)
        self.value_macro_combine = torch.zeros(num_steps + 1, num_processes, 1) # 额外的value_macro, 每一个itr清零一次
        self.value_macro_cat = torch.zeros(num_steps + 1, int(self.args.macro_num/self.gen_batch), self.gen_batch, 1)
        self.actions_high_cat = torch.zeros(num_steps, int(self.args.macro_num/self.gen_batch), self.gen_batch, 1).long()
        self.actions_low_cat = torch.zeros(num_steps, int(self.args.macro_num/self.gen_batch), self.gen_batch, 1).long()
        self.returns_low = torch.zeros(num_steps + 1, num_processes, 1)
        self.returns_high = torch.zeros(num_steps + 1, num_processes, 1)
        self.action_log_probs_low = torch.zeros(num_steps, num_processes, 1)
        self.action_log_probs_high = torch.zeros(num_steps, num_processes, 1)
        self.actions_low = torch.zeros(num_steps, num_processes, 1).long()
        self.actions_high = torch.zeros(num_steps, num_processes, 1).long()
        self.uncertainty = torch.zeros(num_steps, num_processes, 1).long()
        self.state_high_macro = [[] for _ in range(int(self.args.macro_num/self.gen_batch))]
        self.state_high = []
        self.state_low = []
        self.length = torch.zeros(num_steps, num_processes, 1)

        self.masks = torch.zeros(num_steps + 1, num_processes, 1)
        self.num_steps = num_steps
        self.step = 0
        self.Dict_hour = {}

        for i in range(24):
            if i>=0 and i<6:
                self.Dict_hour[i] = 0
            elif i>=6 and i<12:
                self.Dict_hour[i] = 1
            elif i>=12 and i<18:
                self.Dict_hour[i] = 2
            else:
                self.Dict_hour[i] = 3

 
    def after_update(self, level):
        '''
        level=0: micro update
        level=1: macro update
        level=2: cooperative update
        '''
        if level==0 or level==2:
            self.rewards_high = torch.zero_(self.rewards_high)
            self.rewards_low = torch.zero_(self.rewards_low)
            self.value_preds_high = torch.zero_(self.value_preds_high)
            self.value_preds_low = torch.zero_(self.value_preds_low)
            self.returns_low = torch.zero_(self.returns_low)
            self.returns_high = torch.zero_(self.returns_high)
            self.action_log_probs_low = torch.zero_(self.action_log_probs_low)
            self.action_log_probs_high = torch.zero_(self.action_log_probs_high)
            self.actions_low = torch.zero_(self.actions_low)
            self.actions_high = torch.zero_(self.actions_high)
            self.uncertainty = torch.zero_(self.uncertainty)
            self.value_macro_combine = torch.zero_(self.value_macro_combine)
            self.state_high = []
            self.state_low = []
            self.masks = torch.zero_(self.masks) ### 为啥都是1呢？
            self.step = 0
            self.length = torch.zero_(self.length)

        if level==1:
            self.value_macro_cat = torch.zero_(self.value_macro_cat)
            self.actions_high_cat = torch.zero_(self.actions_high_cat)
            self.actions_low_cat = torch.zero_(self.actions_low_cat)
            self.state_high_macro = [[] for _ in range(int(self.args.macro_num/self.gen_batch))]
            self.state_save = None


    def insert(self, uncertainty, state_high, state_low, actions_high, actions_low, action_log_probs_high, action_log_probs_low, value_preds_high, value_preds_low, value_macro, length, masks, epoch):
        self.state_high.append(state_high)
        self.state_low.append(state_low)
        self.actions_high[self.step].copy_(actions_high.unsqueeze(dim=1))
        self.actions_low[self.step].copy_(actions_low.unsqueeze(dim=1))
        self.action_log_probs_high[self.step].copy_(action_log_probs_high.unsqueeze(dim=1))
        self.action_log_probs_low[self.step].copy_(action_log_probs_low.unsqueeze(dim=1))
        self.value_preds_high[self.step].copy_(value_preds_high.unsqueeze(dim=1))
        self.value_preds_low[self.step].copy_(value_preds_low.unsqueeze(dim=1))
        self.value_macro_combine[self.step].copy_(value_macro.unsqueeze(dim=1))
        self.uncertainty[self.step].copy_(uncertainty.unsqueeze(dim=1))
        self.masks[self.step].copy_(masks.unsqueeze(dim=1))
        self.length[self.step].copy_(length.unsqueeze(dim=1))
        self.step += 1



    def compute_returns(self,next_value_high, next_value_low, next_value_macro = None):

        self.value_preds_high[-1] = next_value_high.unsqueeze(dim=1)
        self.value_preds_low[-1] = next_value_low.unsqueeze(dim=1)
        if next_value_macro is not None:
            self.value_macro_combine[-1] = next_value_macro.unsqueeze(dim=1)

        gae = 0
        for step in reversed(range(self.rewards_high.size(0))):
            if next_value_macro is not None:
                delta = (1-self.args.macro_coef) * (self.rewards_high[step] + self.args.gamma * self.value_preds_high[step + 1] * self.masks[step + 1] - self.value_preds_high[step]) + self.args.macro_coef * self.value_macro_combine[step]
            else:
                delta = self.rewards_high[step] + self.args.gamma * self.value_preds_high[step + 1] * self.masks[step + 1] - self.value_preds_high[step]
            gae = delta + self.args.gamma * self.args.gae_lambda * self.masks[step + 1] * gae
            self.returns_high[step] = gae + self.value_preds_high[step]

        gae = 0
        for step in reversed(range(self.rewards_low.size(0))):
            delta = self.rewards_low[step] + self.args.gamma * self.value_preds_low[step + 1] * self.masks[step + 1] - self.value_preds_low[step]
            gae = delta + self.args.gamma * self.args.gae_lambda * self.masks[step + 1] * gae
            self.returns_low[step] = gae + self.value_preds_low[step]

    def micro2macro(self, real_OD_origin, model_path, init_prob, region2loc, loc2region, macro_critic, actor_critic, device, itr):

        batch = self.gen_batch

        print('generate macro data!')

        with torch.no_grad():
            
            for epoch in tqdm(range(int(self.args.macro_num / batch))):

                state1, state2, state = init_loc(self.args, init_prob, region2loc, loc2region, device, batch)

                length = [1] * batch

                for step in range(self.args.num_steps):
                    
                    # Sample actions
                    value_high, action_high, action_log_probs_high, uncertainty, value_low, action_low, action_log_probs_low = actor_critic.act(state,length,length)

                    value_macro = macro_critic.get_macro_value((state1, action_high), length)

                    self.value_macro_cat[step, epoch % int(self.args.macro_num/batch)].copy_(value_macro.detach().cpu().unsqueeze(dim=1))
                    self.actions_high_cat[step, epoch % int(self.args.macro_num/batch)].copy_(action_high.detach().cpu().unsqueeze(dim=1))
                    self.actions_low_cat[step, epoch % int(self.args.macro_num/batch)].copy_(action_low.detach().cpu().unsqueeze(dim=1))

                    # state update
                    state1 = torch.cat((state1, action_high.unsqueeze(dim=1)),dim=1)
                    state2 = torch.cat((state2, action_low.unsqueeze(dim=1)),dim=1)
                    state = (state1.to(device), state2.to(device))

                    length = [step + 2] * batch

                state1 = state1.detach().cpu().numpy().tolist()
                state2 = state2.detach().cpu().numpy().tolist()

        real_OD = (real_OD_origin * self.args.macro_num / self.args.user_num).reshape(-1, self.args.total_regions, self.args.total_regions)

        self.value_macro = self.value_macro_cat.view(self.num_steps + 1, -1, 1)
        self.actions_low_macro = self.actions_low_cat.view(self.num_steps, -1, 1)
        self.actions_high_macro = self.actions_high_cat.view(self.num_steps, -1, 1)

        self.state_save = {}
        self.value_macro_save = torch.zeros([self.args.num_days * int(24/self.args.hour_agg), self.args.total_regions, self.args.total_regions])
        self.OD_flow = torch.zeros([self.args.num_days*int(24/self.args.hour_agg), self.args.total_regions, self.args.total_regions])

        num_processes = batch
        

        for step in tqdm(range(self.num_steps-1)):
            for index_a in range(self.value_macro.shape[1]):
                if self.actions_low_macro[step+1,index_a]!=self.actions_low_macro[step,index_a]:
                    flow_id = (step+1) // self.args.hour_agg
                    if flow_id not in self.state_save:
                        self.state_save[flow_id] = {}
                    o,d = self.actions_high_macro[step,index_a].item(), self.actions_high_macro[step+1,index_a].item()
                    self.OD_flow[flow_id,o,d] += 1
                    if o not in self.state_save[flow_id]:
                        self.state_save[flow_id][o] = {d:[]}
                    if d not in self.state_save[flow_id][o]:
                        self.state_save[flow_id][o][d] = []
                    self.state_save[flow_id][o][d].append(self.actions_high_macro[:step+1,index_a].squeeze(dim=1).numpy().tolist())
                    self.value_macro_save[flow_id, o,d] += self.value_macro[step,index_a].item()


        if self.args.value_agg == 'mean':
            self.value_macro_save /= self.OD_flow + 1e-5

        self.index_save = {}

        print('save file...')

        for flow in tqdm(list(self.state_save.keys())):
            for o in self.state_save[flow]:
                for d in self.state_save[flow][o]:
                    self.index_save[(flow,o,d)] = 1
                    f = open(model_path+'state_save/state_save_{}_{}_{}.pkl'.format(flow,o,d),'wb')
                    pickle.dump(self.state_save[flow][o][d],f)

        del self.state_save

        if self.args.loss_type == 'error':
            self.reward_save = real_OD - self.OD_flow # 42 * num_regions * num_regions
        elif self.args.loss_type == 'relative':
            self.reward_save = (real_OD - self.OD_flow)/(real_OD+1)
        elif self.args.loss_type == 'abs-error':
            self.reward_save = -torch.abs(real_OD - self.OD_flow)
        elif self.args.loss_type == 'abs-relative':
            self.reward_save = -torch.abs((real_OD - self.OD_flow)/(real_OD+1))
        elif self.args.loss_type == 'sigmoid-relative':
            self.reward_save = (torch.sigmoid((real_OD - self.OD_flow)/(real_OD+1))+1e-6).log()
        elif self.args.loss_type == 'sigmoid-abs':
            self.reward_save = (torch.sigmoid(real_OD - self.OD_flow)+1e-6).log()
        next_value_macro = torch.zeros([1, self.args.total_regions, self.args.total_regions])
        self.value_macro_save = torch.cat((self.value_macro_save, next_value_macro), dim=0) # 43 * num_regions * num_regions

        self.reward_record = real_OD - self.OD_flow

        index = torch.argmin(self.reward_record).item()
        flow, o, d = int(index // (self.args.total_regions)**2), int(index % (self.args.total_regions)**2 // self.args.total_regions), int(index % self.args.total_regions)
        

        self.last_min_index = (flow, o, d)

        index = torch.argmax(self.reward_record).item()
        flow, o, d = int(index // (self.args.total_regions)**2), int(index % (self.args.total_regions)**2 // self.args.total_regions), int(index % self.args.total_regions)
        

        self.last_max_index = (flow, o, d)


    def feed_forward_value(self, mini_batch_size, model_path,first = True):

        mini_batch_size = min(len(self.index_save), mini_batch_size)

        self.sampler = BatchSampler(list(self.index_save.keys()), mini_batch_size, drop_last = False)

        for indice_index, indices in enumerate(self.sampler):
            state_batch, action_batch, return_batch = [], [], []
            for ind in indices:
                f = open(model_path+'state_save/state_save_{}_{}_{}.pkl'.format(ind[0], ind[1], ind[2]),'rb')
                temp = pickle.load(f)
                state_batch.append(temp)
                action_batch.append(ind[2])
                return_batch.append(self.reward_save[ind[0]][ind[1]][ind[2]])

            max_len = max([len(s) for s in state_batch])

            action_batch = [[action] * max_len for action in action_batch]
            
            for index, s in enumerate(state_batch):
                for m in s:
                    m += [self.args.total_regions] * (self.num_steps-len(m))
                s += [[self.args.total_regions] * (self.num_steps) for _ in range(max_len-len(s))]


            state_batch = torch.tensor(state_batch).long()
            action_batch = torch.tensor(action_batch).long()
            return_batch = torch.tensor(return_batch)

            # batch_size * max_len * 335
            yield return_batch, action_batch, state_batch

       


    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None, level=0, drop=True):

        num_steps, num_processes = self.rewards_high.size()[0:2]
        batch_size = num_processes * num_steps
        
        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(num_processes, num_steps, num_processes * num_steps,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        sampler = BatchSampler(SubsetRandomSampler(range(batch_size)),mini_batch_size,drop_last=drop)

        assert level in [0,1]
        for indices in sampler:
            if level==0:
                value_preds_batch = self.value_preds_high[:-1].view(-1, 1)[indices]
                return_batch = self.returns_high[:-1].view(-1, 1)[indices]
                masks_batch = self.masks.view(-1, 1)[indices]
                old_action_log_probs_batch = self.action_log_probs_high.view(-1,1)[indices]
                
            else:
                value_preds_batch = self.value_preds_low[:-1].view(-1, 1)[indices]
                return_batch = self.returns_low[:-1].view(-1, 1)[indices]
                masks_batch = self.masks.view(-1, 1)[indices]
                old_action_log_probs_batch = self.action_log_probs_low.view(-1,1)[indices]

            actions_high_batch = self.actions_high.view(-1,1)[indices].squeeze(dim=1)
            actions_low_batch = self.actions_low.view(-1,1)[indices].squeeze(dim=1)
            length_origin = self.length.view(-1,1)[indices].squeeze(dim=1)

            uncertainty = self.uncertainty.view(-1,1)[indices]
            adv_targ = None if advantages is None else advantages.view(-1, 1)[indices]

            indices = [(ind//num_processes, ind % num_processes) for ind in indices]

            state1 = [self.state_high[ind[0]][ind[1]] for ind in indices]
            state2 = [self.state_low[ind[0]][ind[1]] for ind in indices]

            # padding
            max_len = max([len(i) for i in state1])

            length = [len(i) for i in state1]

            state1 = torch.cat([F.pad(s,(0, max_len-len(s)), mode='constant', value=self.args.total_regions).unsqueeze(dim=0) for s in state1], dim=0).long()
            state2 = torch.cat([F.pad(s,(0, max_len-len(s)), mode='constant', value=self.args.total_locations).unsqueeze(dim=0) for s in state2], dim=0).long()

            yield actions_high_batch, actions_low_batch, value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ, uncertainty, state1, state2, length, length_origin
