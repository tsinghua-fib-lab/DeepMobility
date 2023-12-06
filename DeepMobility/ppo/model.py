import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ppo.distributions import Bernoulli, Categorical, DiagGaussian
from ppo.utils import init
import pandas as pd


class Factorization_loc(nn.Module):
    def __init__(self, args):
        super(Factorization_loc, self).__init__()
        self.args = args
        self.V_loc = nn.Embedding(args.total_locations+1, args.embedding_dim, padding_idx = args.total_locations)
        
    def forward(self,loc_id1, loc_id2):
        return torch.sum(self.V_loc(loc_id1) * self.V_loc(loc_id2),dim=1)


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


class Embedding_Layer(nn.Module):
    def __init__(self, args, loc_emb=None, region_emb=None, device=None):
        super(Embedding_Layer, self).__init__()

        self.embedding_region = nn.Embedding(args.total_regions+1, args.embedding_dim, padding_idx = args.total_regions)
        self.embedding_loc = nn.Embedding(args.total_locations+1, args.embedding_dim, padding_idx = args.total_locations)

        self.GRU_region = nn.GRU(args.embedding_dim, args.hidden_size, num_layers = 1, batch_first=True)
        self.GRU_loc = nn.GRU(args.embedding_dim * 2, args.hidden_size, num_layers = 1, batch_first=True)
        self.t_emb = nn.Embedding(int(24/args.hour_agg)*args.num_days, args.embedding_dim)
        self.t_enc = nn.Embedding(24, args.embedding_dim)
        if args.is_week:
            if args.long_days==0:
                self.week_enc = nn.Embedding(args.num_days, int(args.embedding_dim/2))
            else:
                self.week_enc = nn.Embedding(7, int(args.embedding_dim/2))


class Macro_critic(nn.Module):
    def __init__(self, args, emb, device):
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))
        super(Macro_critic, self).__init__()
        self.args = args
        self.device = device
        self.t_enc = emb.t_enc
        self.emb_region = emb.embedding_region
        self.state_emb = emb.GRU_region
        if not args.is_week:
            self.critic = MLP(args.hidden_size + 2 * args.embedding_dim, args.hidden_size, 1, num_layers=args.macro_num_layers)
        else:
            self.critic = MLP(args.hidden_size + 2 * args.embedding_dim + int(0.5 * args.embedding_dim), args.hidden_size, 1, num_layers=args.macro_num_layers)
            self.week_enc = emb.week_enc

    def get_macro_value(self, inputs, length, length_reduce=None):

        if self.args.history_div != 0:
            assert length_reduce is not None

        else:
            length_reduce = length

        state, action = inputs

        x = self.emb_region(state)

        packed = nn.utils.rnn.pack_padded_sequence(x, length_reduce, batch_first=True, enforce_sorted=False)
        _, h_n = self.state_emb(packed)
        emb_state = h_n.squeeze(dim=0)
        t_emb = self.t_enc(((torch.tensor(length) + 1) % 24).long().to(self.device))
        if self.args.is_week:
            week_emb = self.week_enc(((torch.tensor(length)-1 )// 24).long().to(self.device))

        a = self.emb_region(action)

        if self.args.macro_detach:
            emb_state = emb_state.detach()
            t_emb =  t_emb.detach()
            a = a.detach()
            if self.args.is_week:
                week_emb = week_emb.detach()
        if not self.args.is_week:
            value = self.critic(torch.cat((emb_state,t_emb, a),dim=-1)).squeeze(dim=1)
        else:
            value = self.critic(torch.cat((emb_state,t_emb, week_emb, a),dim=-1)).squeeze(dim=1)
        return value


    def evaluate(self, inputs):

        state, action = inputs
        batch_size, max_len = state.shape[0], state.shape[1]

        state = state.view(-1, state.shape[-1])
        action = action.view(-1)

        length = torch.sum(state!=self.args.total_regions, dim=-1)
        mask = (length!=0)

        length = length.detach().cpu()

        if torch.sum(mask)==0:
            return None

        length[length==0] += 1 # length cannot equal 0

        value = self.get_macro_value((state, action), length)

        value = value.view(batch_size, max_len)
        mask = mask.view(batch_size, max_len)

        assert value.shape == mask.shape 

        value = torch.sum(value * mask.float(), dim = -1) 

        if self.args.value_agg == 'mean':
            num = torch.sum(mask, dim=-1) + 1e-9
            assert value.shape == num.shape
            value = value / num
        return value.unsqueeze(dim=1)


class DeepMobility(nn.Module):
    def __init__(self,args, emb, region2loc, d_feature, loclonlat, device, history_len=1e9):
        super(DeepMobility, self).__init__()
        history_len = int(history_len)
        self.args = args
        self.policy_high = Policy_High(args, emb, d_feature, device, history_len).to(device)
        self.policy_low = Policy_Low(args, emb, region2loc, loclonlat, device, history_len).to(device)
        self.t_enc = emb.t_enc

    def forward(self):
        raise NotImplementedError

    def act(self, inputs,length, length_reduce=None):

        if self.args.history_div!=0:
            assert length_reduce is not None

        else:
            length_reduce = length

        seq1, seq2 = inputs # N * L, N * L
        value_high, action_high, action_log_probs_high, uncertainty = self.policy_high.act(seq1,length, length_reduce)
        value_low, action_low, action_log_probs_low = self.policy_low.act((seq1, seq2, action_high),length, length_reduce)
        return value_high, action_high, action_log_probs_high, uncertainty, value_low, action_low, action_log_probs_low 

    def evaluate_actions(self, inputs, action, act_label, length):
        raise NotImplementedError


class Policy_High(nn.Module):
    def __init__(self, args, emb, d_feature, device, history_len=1e9):
        super(Policy_High, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))
        
        self.device = device
        self.args = args
        self.history_len = history_len

        self.state_emb = emb.GRU_region
        self.t_enc = emb.t_enc

        self.softmax = nn.LogSoftmax(dim=-1)
        self.emb = emb.embedding_region # D * d_feature

        if not args.is_week:
            self.critic = MLP(args.hidden_size + args.embedding_dim, args.hidden_size, args.num_heads, num_layers=3).to(device)
            self.actor_individual = MLP(args.hidden_size + args.embedding_dim, args.hidden_size, args.total_regions, num_layers=2).to(device)
        else:
            self.critic = MLP(args.hidden_size + args.embedding_dim  + int(0.5 * args.embedding_dim), args.hidden_size, args.num_heads, num_layers=3).to(device)
            self.actor_individual = MLP(args.hidden_size + args.embedding_dim + int(0.5 * args.embedding_dim), args.hidden_size, args.total_regions, num_layers=2).to(device)
            self.week_enc = emb.week_enc

        self.actor_others = DeepGravity(args, emb, device, d_feature).to(device)


    def forward(self, inputs, length):
        '''
        inputs: N * T, region-level trajector, for pretrain
        '''
        x = self.emb(inputs.long())

        t_emb = self.t_enc((length % 24).long().to(self.device)) # has already plus 1
        if self.args.is_week:
            week_emb = self.week_enc(((torch.tensor(length)-2) // 24 % 7).long().to(self.device))
        emb_state, _ = self.state_emb(x)

        if not self.args.is_week:
            pred = self.softmax(self.actor_individual(torch.cat((emb_state,t_emb),dim=-1)))
        else:
            
            pred = self.softmax(self.actor_individual(torch.cat((emb_state,t_emb, week_emb),dim=-1)))

        return pred


    def act(self, inputs, length, length_reduce):


        '''
        inputs: N * T, region-level trajectory
        
        '''
        if self.args.history_div != 0:
            assert length_reduce is not None

        x = self.emb(inputs[:,-self.history_len:])

        packed = nn.utils.rnn.pack_padded_sequence(x, length_reduce, batch_first=True, enforce_sorted=False)
        _, h_n = self.state_emb(packed)

        emb_state = h_n.squeeze(dim=0) # N * hidden_size

        t_emb = self.t_enc(((torch.tensor(length)+1) % 24).long().to(self.device))

        if self.args.is_week:
            week_emb = self.week_enc(((torch.tensor(length) -1)// 24 % 7).long().to(self.device))

        if not self.args.is_week:
            value_multi = self.critic(torch.cat((emb_state,t_emb),dim=-1)) # N * (hidden_size+embedding_dim)
        else:
            value_multi = self.critic(torch.cat((emb_state,t_emb, week_emb),dim=-1)) # N * (hidden_size+embedding_dim)

        uncertainty = torch.var(value_multi, dim=-1, unbiased=False, keepdim=False) # N #limit in [0,1]
        value = torch.mean(value_multi,dim=-1)
        mask_uncertainty = (uncertainty>self.args.uncertainty).float() # 0: individual, 1: gravity

        # individual preference
        if not self.args.is_week:
            act_prob1 = F.softmax(self.actor_individual(torch.cat((emb_state,t_emb),dim=-1)),dim=-1)
        else:
            act_prob1 = F.softmax(self.actor_individual(torch.cat((emb_state,t_emb, week_emb),dim=-1)),dim=-1)

        if torch.isnan(act_prob1).any() or (act_prob1<0).any():
            print(torch.isnan(act_prob1).any(), (act_prob1<0).any())
            exit()

        dist1 = torch.distributions.Categorical(act_prob1)
        action1 = dist1.sample()
        action1_log_probs = dist1.log_prob(action1)

        # social interaction
        t = torch.tensor([inputs.shape[1]// self.args.hour_agg % (self.args.num_days * 24 / self.args.hour_agg)]*inputs.shape[0]).unsqueeze(dim=1).to(self.device).long()
        out= F.relu(self.actor_others(inputs[:,-1].unsqueeze(dim=1),t)) + 1e-7
        act_prob2 = out / torch.sum(out,dim=-1,keepdim=True)
        dist2 = torch.distributions.Categorical(act_prob2)
        action2 = dist2.sample()
        action2_log_probs = dist2.log_prob(action2)
        
        assert mask_uncertainty.shape == action1.shape
        assert mask_uncertainty.shape == action2.shape

        assert mask_uncertainty.shape == action1_log_probs.shape
        assert mask_uncertainty.shape == action2_log_probs.shape

        action = ((1-mask_uncertainty) * action1 + mask_uncertainty * action2).long()
        action_log_probs = (1-mask_uncertainty)  * action1_log_probs + mask_uncertainty * action2_log_probs

        return value, action, action_log_probs, mask_uncertainty


    def get_value(self, inputs, length, length_reduce=None):


        if self.args.history_div != 0:
            assert length_reduce is not None

        x = self.emb(inputs.long()[:,-self.history_len:])

        packed = nn.utils.rnn.pack_padded_sequence(x, length_reduce, batch_first=True, enforce_sorted=False)
        test, h_n = self.state_emb(packed)
        emb_state = h_n.squeeze(dim=0) # N * hidden_size

        t_emb = self.t_enc(((torch.tensor(length) + 1) % 24).long().to(self.device))
        if self.args.is_week:
            week_emb = self.week_enc(((torch.tensor(length)-1) // 24).long().to(self.device))
            
        if not self.args.is_week:
            value_multi = self.critic(torch.cat((emb_state,t_emb),dim=-1)) # N * num_heads
        else:
            value_multi = self.critic(torch.cat((emb_state,t_emb, week_emb),dim=-1)) # N * num_heads
        value = torch.mean(value_multi,dim=-1) # N

        return value


    def evaluate_actions(self, inputs, action, act_label, length, length_reduce=None):

        '''
        inputs: N * T, region-level trajectory
        action: N * 1
        act_label: N * 1, 0/1, 0 represents individual, 1 represents gravity
        length: sequence length of states
        '''
        if self.args.history_div != 0:
            assert length_reduce is not None

        else:
            length_reduce = length

        x = self.emb(inputs.long()[:,-self.history_len:])

        packed = nn.utils.rnn.pack_padded_sequence(x, length_reduce, batch_first=True, enforce_sorted=False)
        test, h_n = self.state_emb(packed)
        emb_state = h_n.squeeze(dim=0) # N * hidden_size

        t_emb = self.t_enc(((length.clone() + 1) % 24).long().to(self.device))
        if self.args.is_week:
            week_emb = self.week_enc(((length.clone()-1) // 24).long().to(self.device))

        if not self.args.is_week:
            value_multi = self.critic(torch.cat((emb_state,t_emb),dim=-1)) # N * num_heads
        else:
            value_multi = self.critic(torch.cat((emb_state,t_emb, week_emb),dim=-1)) # N * num_heads
        value = torch.mean(value_multi,dim=-1,keepdim=True) # N * 1

        # individual preference
        if not self.args.is_week:
            act_prob1 = F.softmax(self.actor_individual(torch.cat((emb_state,t_emb),dim=-1)),dim=-1)
        else:
            act_prob1 = F.softmax(self.actor_individual(torch.cat((emb_state,t_emb, week_emb),dim=-1)),dim=-1)
        
        dist1 = torch.distributions.Categorical(act_prob1)
        dist1_entropy = dist1.entropy().unsqueeze(dim=1)

        # social interaction
        t = (length.clone() // self.args.hour_agg  % (self.args.num_days * 24 / self.args.hour_agg) ).unsqueeze(dim=1).long().to(self.device)

        out = F.relu(self.actor_others(inputs[torch.arange(inputs.shape[0]),(length_reduce.clone()-1).long()].unsqueeze(dim=1),t))+1e-7
        act_prob2 = out / torch.sum(out,dim=-1,keepdim=True)
        dist2 = torch.distributions.Categorical(act_prob2)
        dist2_entropy = dist2.entropy().unsqueeze(dim=1)

        assert act_label.shape == dist1_entropy.shape
        assert act_label.shape == dist2_entropy.shape

        if self.args.detach:
            dist_entropy = ((1-act_label) * dist1_entropy + (act_label * dist2_entropy).detach()).mean()

        else:
            dist_entropy = ((1-act_label) * dist1_entropy + act_label * dist2_entropy).mean()

        act_log_prob1 = dist1.log_prob(action).unsqueeze(dim=1)
        act_log_prob2 = dist2.log_prob(action).unsqueeze(dim=1)

        assert act_log_prob1.shape == act_label.shape
        assert act_log_prob2.shape == act_label.shape

        if self.args.detach:
            action_log_probs = act_log_prob1 * (1-act_label) + (act_log_prob2 * act_label).detach()

        else:
            action_log_probs = act_log_prob1 * (1-act_label) + act_log_prob2 * act_label

        return value, action_log_probs, dist_entropy


class Policy_Low(nn.Module):
    def __init__(self, args, emb, region2loc, loclonlat, device, history_len):
        '''
        region2loc: N1 * N2, N1 is the number of regions, N2 is the number of corresponding locations, max_N2 = 
        '''
        super(Policy_Low, self).__init__()

        self.history_len = history_len

        self.region2loc = region2loc

        self.device = device

        self.t_enc = emb.t_enc

        self.loclonlat = loclonlat

        self.args = args

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.emb_region = emb.embedding_region
        self.emb_loc = emb.embedding_loc

        self.emb_region = emb.embedding_region
        self.emb_loc = emb.embedding_loc

        self.state_emb = emb.GRU_loc

        self.critic = MLP(args.hidden_size, args.hidden_size, 1, num_layers=3).to(device)

        if not args.is_week:
            self.state_proj = MLP(args.hidden_size + 2*args.embedding_dim, args.hidden_size, args.hidden_size, num_layers = 2).to(device)
        else:
            self.state_proj = MLP(args.hidden_size + 2*args.embedding_dim + int(1/2*args.embedding_dim), args.hidden_size, args.hidden_size, num_layers = 2).to(device)
            self.week_enc = emb.week_enc

        self.proj = nn.Sequential(nn.Linear(args.embedding_dim,args.hidden_size), nn.Tanh())
        self.softmax = nn.LogSoftmax(dim=-1)


    def act(self, inputs, length, length_reduce=None):
        '''
        inputs: (state1, state2, high-level action)
        state1: N * T, region-level trajectory
        state2: N * T, location_level trajectory
        high-level action: N
        mask: 0/1, 1 denotes the position to be masked

        '''
        if self.args.history_div != 0:
            assert length_reduce is not None


        s1, s2, action_high = inputs

        x1, x2 = self.emb_region(s1)[:,-self.history_len:], self.emb_loc(s2)[:,-self.history_len:]

        x = torch.cat((x1,x2),dim=-1)

        packed = nn.utils.rnn.pack_padded_sequence(x, length_reduce, batch_first=True, enforce_sorted=False)

        _, h_n = self.state_emb(packed)

        act_emb = self.emb_region(action_high)

        t_emb = self.t_enc(((torch.tensor(length)+1) % 24).long().to(self.device))
        if self.args.is_week:
            week_emb = self.week_enc(((torch.tensor(length)-1) // 24).long().to(self.device))

        if not self.args.is_week:
            emb_s = self.state_proj(torch.cat((h_n.squeeze(dim=0),act_emb,t_emb),dim=-1)) # N * (hidden_size + embedding_dim)s
        else:
            emb_s = self.state_proj(torch.cat((h_n.squeeze(dim=0),act_emb,t_emb, week_emb),dim=-1)) # N * (hidden_size + embedding_dim)s

        value = self.critic(emb_s).squeeze(dim=1)

        # attention mechanism
        candidate_loc = self.emb_loc(self.region2loc.to(self.device)[action_high.long()]) # N * N2 * embedding_size
        
        
        if self.args.select_loc == 'Attn':
            H = self.proj(candidate_loc) # (N * L) * N2 * hidden_size
            score = H.matmul(emb_s.unsqueeze(dim=2)).squeeze(dim=2) # (N * L) * N2
            mask = (self.region2loc.to(self.device)[action_high.long()]==self.args.total_locations)
            score = score.masked_fill(mask, value=-1e10)
            weight = F.softmax(score, dim=-1)
        
        else:
            emb_s = emb_s.unsqueeze(dim=1).repeat(1,candidate_loc.shape[1],1)
            candidate_loc = self.cand(candidate_loc)

            if self.args.loc_distance:
                d_loc = self.loclonlat[self.region2loc.to(self.device)[action_high.long()]].to(self.device)
                o_loc = self.loclonlat[s2[:,-1:].repeat(1,d_loc.shape[1])].to(self.device)
                distance = torch.sqrt(torch.sum((o_loc-d_loc).pow(2),dim=-1, keepdim=True)) # N * D * 1
                distance = torch.clip(distance/0.01,max=49).long()
                distance = self.dis_emb(distance).squeeze(dim=2)
                candidate_loc = torch.cat((distance, candidate_loc,emb_s),dim=-1)

            else:
                candidate_loc = torch.cat((candidate_loc,emb_s),dim=-1)
            score = self.actor_individual(candidate_loc).squeeze(dim=-1)
            mask = (self.region2loc.to(self.device)[action_high.long()]==self.args.total_locations)
            score = score.masked_fill(mask, value=-1e10)
            weight = F.softmax(score,dim=-1)

        dist = torch.distributions.Categorical(weight)
        action = dist.sample()

        action_log_probs = dist.log_prob(action)

        action = self.region2loc.to(self.device)[action_high.long()][torch.arange(action.shape[0]),action]

        return value, action, action_log_probs


    def get_value(self, inputs, length, length_reduce=None):

        s1, s2, action_high = inputs

        if self.args.history_div != 0:
            assert length_reduce is not None

        x1, x2 = self.emb_region(s1)[:,-self.history_len:], self.emb_loc(s2)[:,-self.history_len:]

        x = torch.cat((x1,x2),dim=-1)

        packed = nn.utils.rnn.pack_padded_sequence(x, length_reduce, batch_first=True, enforce_sorted=False)

        _, h_n = self.state_emb(packed)

        act_emb = self.emb_region(action_high)

        t_emb = self.t_enc(((torch.tensor(length)+1) % 24).long().to(self.device))
        if self.args.is_week:
            week_emb = self.week_enc(((torch.tensor(length)-1) // 24).long().to(self.device))

        if not self.args.is_week:
            emb_s = self.state_proj(torch.cat((h_n.squeeze(dim=0),act_emb, t_emb),dim=-1)) # N * (hidden_size + embedding_dim)s
        else:
            emb_s = self.state_proj(torch.cat((h_n.squeeze(dim=0),act_emb, t_emb, week_emb),dim=-1)) # N * (hidden_size + embedding_dim)s

        value = self.critic(emb_s).squeeze(dim=1)

        return value


    def evaluate_actions(self, inputs, action_low, length, length_reduce=None):

        '''
        action_low: N
        '''

        s1, s2, action_high = inputs

        if self.args.history_div != 0:
            assert length_reduce is not None
        
        x1, x2 = self.emb_region(s1)[:,-self.history_len:], self.emb_loc(s2)[:,-self.history_len:]

        x = torch.cat((x1,x2),dim=-1)

        packed = nn.utils.rnn.pack_padded_sequence(x, length_reduce, batch_first=True, enforce_sorted=False)

        _, h_n = self.state_emb(packed)

        act_emb = self.emb_region(action_high)

        t_emb = self.t_enc(((torch.tensor(length)+1) % 24).long().to(self.device))
        if self.args.is_week:
            week_emb = self.week_enc(((torch.tensor(length)-1) // 24).long().to(self.device))

        if not self.args.is_week:
            emb_s = self.state_proj(torch.cat((h_n.squeeze(dim=0),act_emb, t_emb),dim=-1)) # N * (hidden_size + embedding_dim)s
        else:
            emb_s = self.state_proj(torch.cat((h_n.squeeze(dim=0),act_emb, t_emb, week_emb),dim=-1)) # N * (hidden_size + embedding_dim)s

        value = self.critic(emb_s)

        candidate_loc = self.emb_loc(self.region2loc.to(self.device)[action_high.long()]) # N * N2 * embedding_size

        if self.args.select_loc == 'Attn':
            H = self.proj(candidate_loc) # N * N2 * hidden_size
            score = H.matmul(emb_s.unsqueeze(dim=2)).squeeze(dim=2) # (N * L) * N2
            mask = (self.region2loc.to(self.device)[action_high.long()]==self.args.total_locations)
            score = score.masked_fill(mask, value=-1e10)
            weight = F.softmax(score, dim=-1)
        
        else:
            emb_s = emb_s.unsqueeze(dim=1).repeat(1,candidate_loc.shape[1],1)
            candidate_loc = self.cand(candidate_loc)
            if self.args.loc_distance:
                d_loc = self.loclonlat[self.region2loc.to(self.device)[action_high.long()]].to(self.device)
                o_loc = self.loclonlat[s2[torch.arange(s2.shape[0]),torch.tensor(length).long()-1].unsqueeze(dim=1).repeat(1,d_loc.shape[1])].to(self.device)
                distance = torch.sqrt(torch.sum((o_loc-d_loc).pow(2),dim=-1, keepdim=True)) # N * D * 1
                distance = torch.clip(distance/0.01,max=49).long()
                distance = self.dis_emb(distance).squeeze(dim=2)
                candidate_loc = torch.cat((distance, candidate_loc,emb_s),dim=-1)

            else:
                candidate_loc = torch.cat((candidate_loc,emb_s),dim=-1)

            score = self.actor_individual(candidate_loc).squeeze(dim=-1)
            mask = (self.region2loc.to(self.device)[action_high.long()]==self.args.total_locations)
            score = score.masked_fill(mask, value=-1e10)
            weight = F.softmax(score,dim=-1)

        dist = torch.distributions.Categorical(weight)
        dist_entropy = dist.entropy().mean()

        # action id transformation
        action_low = action_low.unsqueeze(dim=1).expand(action_low.shape[0],weight.shape[-1])
        log_prob = torch.log(weight)
        index = (action_low==self.region2loc.to(self.device)[action_high.long()])

        _, action_reindex = torch.where(action_low==self.region2loc.to(self.device)[action_high.long()])

        action_log_probs = dist.log_prob(action_reindex).unsqueeze(dim=1)

        return value, action_log_probs, dist_entropy

class DeepGravity(nn.Module):
    def __init__(self, args, emb, device, d_feature=None):
        super(DeepGravity, self).__init__()
        self.device = device
        self.emb = emb.embedding_region
        self.d = (torch.arange(args.total_regions)).long().to(device)
        self.args = args
        self.distance = d_feature[0] # D * 2
        self.d_feature = d_feature[1] # D * d_feature
        self.gravity = MLP(self.d_feature.shape[1] * 2 + args.embedding_dim + 1, args.hidden_size, 1, num_layers = 15, \
        activation=nn.Tanh() if args.gravity_activation=='tanh' else nn.LeakyReLU(), dropout=True).to(device)
        self.t_emb = emb.t_emb

    def get_features(self,inputs):
        '''
        inputs: region_id
        '''
        return self.distance[inputs].to(self.device), self.d_feature[inputs].to(self.device)


    def forward(self, inputs, t):

        '''
        t: N * 1
        '''

        o_loc, o_feature = self.get_features(inputs) # N * 2, N * d_feature
        N = inputs.shape[0]

        d_loc, d_feature = self.get_features(self.d) # D * 2, D * d_feature
        D = d_loc.shape[0]

        o_loc = o_loc.expand(N, D, o_loc.shape[-1]) # N * D * 2
        o_feature = o_feature.expand(N, D, o_feature.shape[-1]) # N * D * d_feature

        distance = torch.sqrt(torch.sum((o_loc-d_loc).pow(2),dim=-1, keepdim=True)) # N * D * 1

        t_emb = self.t_emb(t).expand(N,D,self.args.embedding_dim) # N * D * embedding_size

        d_feature = d_feature.unsqueeze(dim=0).expand(N, D, d_feature.shape[-1]) # N * D * d_feature

        od_feature = torch.cat((distance, o_feature, d_feature,t_emb), dim=-1) # N * D * (2*d_feature+embedding_dim+1)

        flow = self.gravity(od_feature).squeeze(dim=2) # N * D

        return flow



