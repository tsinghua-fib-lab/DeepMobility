import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from torch import autograd
from ppo.utils import init
import mlflow
from stable_baselines3.common.running_mean_std import RunningMeanStd


class MLP(nn.Module):

    def __init__(self, dim_in,  dim_hidden, dim_out, num_layers=2, activation=nn.Tanh(), dropout=False):
        super(MLP, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.linears = nn.ModuleList()
        self.linears.append(init_(nn.Linear(dim_in, dim_hidden)))

        for _ in range(num_layers-2):
            self.linears.append(init_(nn.Linear(dim_hidden, dim_hidden)))

        self.linears.append(init_(nn.Linear(dim_hidden, dim_out)))

        self.activation = activation

        self.dropout = dropout

        if self.dropout:
            self.dropout_out =  torch.nn.Dropout(0.35)

    def forward(self, x):
        for m in self.linears[:-1]:
            x = self.activation(m(x))
            if self.dropout:
                x = self.dropout_out(x)

        return self.linears[-1](x)

class Discriminator(nn.Module):
    def __init__(self,args, emb, device):

        super(Discriminator, self).__init__()
        self.args = args

        self.hidden_size = args.hidden_size

        self.GRU_high = nn.GRU(args.embedding_dim, self.hidden_size, num_layers = 1, batch_first=True)
        self.GRU_low = nn.GRU(2*args.embedding_dim, self.hidden_size, num_layers = 1, batch_first=True)

        self.emb_high = nn.Embedding(args.total_regions+1, args.embedding_dim, padding_idx = args.total_regions)
        self.emb_low = nn.Embedding(args.total_locations+1, args.embedding_dim, padding_idx = args.total_locations)
        self.t_enc = nn.Embedding(24, args.embedding_dim)
        self.week_enc = nn.Embedding(args.num_days, int(0.5*args.embedding_dim))

        if not args.is_week:
            self.proj_high = MLP(self.hidden_size + 2 * args.embedding_dim, self.hidden_size, 1, num_layers=2).to(device)
            self.proj_low = MLP(self.hidden_size + 2 * args.embedding_dim, self.hidden_size, 1, num_layers=2).to(device)
        else:
            self.proj_high = MLP(self.hidden_size + 2 * args.embedding_dim + int(0.5*args.embedding_dim), self.hidden_size, 1, num_layers=2).to(device)
            self.proj_low = MLP(self.hidden_size + 2 * args.embedding_dim + int(0.5*args.embedding_dim), self.hidden_size, 1, num_layers=2).to(device)

        self.device = device

        self.returns_low, self.returns_high = None, None
        self.ret_rms_low, self.ret_rms_high = RunningMeanStd(shape=()), RunningMeanStd(shape=())



    def forward_high(self, state, action, length, length_reduce):
        
        state_emb = self.emb_high(state) # N * L * embedding_size
        packed = nn.utils.rnn.pack_padded_sequence(state_emb, length_reduce, batch_first=True, enforce_sorted=False)
        _, state_hidden = self.GRU_high(packed)

        act_emb = self.emb_high(action) # N * embedding_size

        t_emb = self.t_enc(((torch.tensor(length) + 1) % 24).long().to(self.device))

            if self.args.is_week:
                week_emb = self.week_enc(((torch.tensor(length) - 1) // 24).long().to(self.device))

        if not self.args.is_week:
            d = self.proj_high(torch.cat([state_hidden.squeeze(dim=0), act_emb, t_emb], dim=-1))
        else:
            d = self.proj_high(torch.cat([state_hidden.squeeze(dim=0), act_emb, t_emb, week_emb], dim=-1))

        return d

    def forward_low(self, state, action, length, length_reduce):
        
        s1, s2 = state

        x1, x2 = self.emb_high(s1), self.emb_low(s2)
        x = torch.cat((x1,x2),dim=-1)

        packed = nn.utils.rnn.pack_padded_sequence(x, length_reduce, batch_first=True, enforce_sorted=False)
        _, state_hidden = self.GRU_low(packed)

        act_emb = self.emb_low(action)

        temp = (torch.tensor(length) + 1) % 24
        t_emb = self.t_enc(((torch.tensor(length) + 1) % 24).long().to(self.device))
        if self.args.is_week:
            week_emb = self.week_enc(((torch.tensor(length) - 1) // 24).long().to(self.device))
    
        if not self.args.is_week:
            d = self.proj_low(torch.cat([state_hidden.squeeze(dim=0), act_emb,t_emb], dim=-1))
        else:
            d = self.proj_low(torch.cat([state_hidden.squeeze(dim=0), act_emb,t_emb, week_emb], dim=-1))

        return d

    def reward_high_test(self,state, action, length, length_reduce):
        with torch.no_grad():
            self.eval()
            d = torch.sigmoid(self.forward_high(state, action, length, length_reduce))
            reward = d.log() - (1 - d).log()

        return reward / np.sqrt(self.ret_rms_high.var[0] + 1e-8)


    def predict_reward_high(self, state, action, length, length_reduce, masks, update_rms=True):
        with torch.no_grad():
            self.eval()
            d = torch.sigmoid(self.forward_high(state, action, length, length_reduce))
            reward = d.log() - (1 - d).log()

            if self.returns_high is None:
                self.returns_high = reward.clone().detach().cpu()

            if update_rms:
                self.returns_high = self.returns_high * masks * self.args.gamma + reward.detach().cpu()
                self.ret_rms_high.update(self.returns_high.cpu().numpy())

        return reward / np.sqrt(self.ret_rms_high.var[0] + 1e-8)

    def predict_reward_low(self, state, action, length, length_reduce, masks, update_rms=True): #### 这里不对，需要改state!!!
        with torch.no_grad():
            self.eval()
            d = torch.sigmoid(self.forward_low(state, action, length, length_reduce))
            reward = d.log() - (1 - d).log()

            if self.returns_low is None:
                self.returns_low = reward.clone().detach().cpu()

            if update_rms:
                self.returns_low = self.returns_low * masks * self.args.gamma + reward.detach().cpu()
                self.ret_rms_low.update(self.returns_low.cpu().numpy())

        return reward / np.sqrt(self.ret_rms_low.var[0] + 1e-8)

    def update(self, expert_loader, rollouts, optimizer = None, level=0):

        self.train()

        assert level in [0,1]

        # 几个batch的协调需要注意一下！

        policy_data_generator = rollouts.feed_forward_generator(None, mini_batch_size=self.args.gail_batch_size,level=0)

        loss_high, loss_low, correct_high, correct_low, num = 0.0, 0.0, 0.0, 0.0, 0.0

        epoch = 0

        for expert_batch, policy_batch in zip(expert_loader, policy_data_generator):

            optimizer.zero_grad()

            policy_action_high, policy_action_low, _, _, _, _, _, _, policy_state_high, policy_state_low, policy_length, policy_length_origin = policy_batch

            expert_state_high, expert_state_low, expert_action_high, expert_action_low, expert_length, expert_length_origin = expert_batch
            expert_length = expert_length.numpy().tolist()

            policy_high_d = self.forward_high(policy_state_high.to(self.device), policy_action_high.to(self.device),policy_length_origin, policy_length)
            policy_low_d = self.forward_low((policy_state_high.to(self.device), policy_state_low.to(self.device)), policy_action_low.to(self.device), policy_length_origin, policy_length)

            expert_d_high = self.forward_high(expert_state_high.to(self.device), expert_action_high.to(self.device), expert_length_origin, expert_length)
            expert_d_low = self.forward_low((expert_state_high.to(self.device),expert_state_low.to(self.device)), expert_action_low.to(self.device), expert_length_origin, expert_length)

            assert policy_high_d.shape == expert_d_high.shape

            expert_high_loss = F.binary_cross_entropy_with_logits(expert_d_high, torch.ones(expert_d_high.size()).to(self.device),reduction='mean') 
            expert_low_loss = F.binary_cross_entropy_with_logits(expert_d_low, torch.ones(expert_d_low.size()).to(self.device),reduction='mean')

            policy_high_loss = F.binary_cross_entropy_with_logits(policy_high_d, torch.zeros(policy_high_d.size()).to(self.device),reduction='mean')
            policy_low_loss = F.binary_cross_entropy_with_logits(policy_low_d, torch.zeros(policy_low_d.size()).to(self.device),reduction='mean')

            gail_high_loss = expert_high_loss + policy_high_loss
            gail_low_loss = expert_low_loss + policy_low_loss

            loss_high += expert_high_loss.item() * expert_d_high.shape[0] + policy_high_loss.item() * policy_high_d.shape[0]
            loss_low += expert_low_loss.item() * expert_d_low.shape[0] + policy_low_loss.item() * policy_low_d.shape[0]

            if torch.isnan(gail_high_loss).any() or torch.isnan(gail_low_loss):
                print('gail loss nan')
                exit()

            preds_policy_high = torch.round(torch.sigmoid(policy_high_d))
            preds_policy_low = torch.round(torch.sigmoid(policy_high_d))

            preds_expert_high = torch.round(torch.sigmoid(expert_d_high))
            preds_expert_low = torch.round(torch.sigmoid(expert_d_low))

            correct_high += torch.eq(preds_policy_high.detach().cpu(), torch.zeros(policy_high_d.size())).float().sum().item() + torch.eq(preds_expert_high.detach().cpu(), torch.ones(expert_d_high.size())).float().sum().item()
            correct_low += torch.eq(preds_policy_low.detach().cpu(), torch.zeros(policy_low_d.size())).float().sum().item() + torch.eq(preds_expert_low.detach().cpu(), torch.ones(expert_d_low.size())).float().sum().item()
            
            num += preds_policy_high.shape[0] + preds_expert_high.shape[0]

            epoch += 1

            gail_high_loss.backward()
            gail_low_loss.backward()

            optimizer.step()
            
        mlflow.log_metric('acc_high',correct_high/num,step=0)
        mlflow.log_metric('acc_low',correct_low/num,step=0)
        mlflow.log_metric('loss_disc_high',loss_high/num,step=0)
        mlflow.log_metric('loss_disc_low',loss_low/num,step=0)

        return correct_high/num, correct_low/num