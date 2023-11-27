import argparse

import torch

def get_args():
    parser = argparse.ArgumentParser(description='RL')

    # network parameters
    parser.add_argument('--embedding_dim', type=int, default=16, help='embedding size')
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden size')
    parser.add_argument('--num_heads',type=int,default=8, help='number of heads to calculate uncertainty')
    parser.add_argument('--uncertainty',type=float,default=1e10, help='')
    parser.add_argument('--time_enc',action='store_true',default=False, help='')
    parser.add_argument('--is_week',action='store_true',default=False, help='')
    parser.add_argument('--loc_emb', type=str, default='id', choices = ['id','transfer'])
    parser.add_argument('--loc_feature', type=str, default='cnt_norm_100', choices = ['cnt_norm_100','cnt_norm_1000'])
    parser.add_argument('--region_feature', type=str, default='population_norm_100', choices = ['population_norm_100','population_norm_1000'])
    parser.add_argument('--select', type=str, default='MLP', choices = ['Attn','MLP'])
    parser.add_argument('--select_loc', type=str, default='Attn', choices = ['Attn','MLP'])
    parser.add_argument('--test_id', type=int, default=399)
    parser.add_argument('--loc_distance',action='store_true',default=False, help='')
    parser.add_argument('--activation', type=str, default='relu', choices=['relu','tanh'])
    parser.add_argument('--city_param', type=str, default='shanghai', required=True)
    parser.add_argument('--pretrain_name', type=str, default='')
    parser.add_argument('--pretrain_name2', type=str, default='')
    parser.add_argument('--training_num', type=int, default=20000)
    parser.add_argument('--long_days', type=int, default=0)
    parser.add_argument('--eval_dist', type=str, default='jsd')

    # configurations
    parser.add_argument('--cuda_id',type=int,default=0, help='')
    parser.add_argument('--dataset',type=str,default='shanghai', choices=['shanghai', 'beijing','shenzhen','Senegal'], required=True)
    parser.add_argument('--total_locations',type=int,default=None, help='number of locations')
    parser.add_argument('--total_regions',type=int,default=None, help='number of regions')
    parser.add_argument('--seed', type=int, default=6, help='random seed (default: 1)')
    parser.add_argument('--run_name', type=str, required=True)
    parser.add_argument('--with_evaluate',action='store_true',default=False,help='')
    parser.add_argument('--model',type=str,default='', help='')
    parser.add_argument('--machine', type=str, required=True)
    parser.add_argument('--resolution', type=str, required=True, choices = ['500','1000','10000'])
    parser.add_argument('--param_resolution', type=str, required=True, choices = ['500','1000','10000'])

    # training parameters
    parser.add_argument('--simulate_batch_size', type=int,default = 128)
    parser.add_argument('--warmup_iter', type=int,default=5)
    parser.add_argument('--num_steps', type=int,default=None)
    parser.add_argument('--num_updates',type=int,default=2000, help='')
    parser.add_argument('--lr', type=float, default=5e-6, help='learning rate (default: 7e-4)')
    parser.add_argument('--lr_pretrain', type=float, default=3e-4, help='learning rate (default: 7e-4)')
    parser.add_argument('--macro_lr', type=float, default=1e-4, help='learning rate (default: 7e-4)')
    parser.add_argument('--disc_lr', type=float, default=1e-5, help='')
    parser.add_argument('--eps', type=float, default=1e-6, help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument('--no_low',action='store_true',default=False,help='disable low-level policy training')
    parser.add_argument('--disc_interval', type=int,default=1)

    # evaluate
    parser.add_argument('--evaluate_epoch', type=int, default=50, help='')
    parser.add_argument('--evaluate_batch', type=int, default=20000, help='')
    parser.add_argument('--total_OD_num', type=int, default=300000, help='')
   
    # ppo
    parser.add_argument('--clip_param',type=float,default=0.2,help='ppo clip parameter (default: 0.2)')
    parser.add_argument('--ppo_epoch', type=int, default=5, help='number of ppo epochs (default: 4)')
    parser.add_argument('--num_mini_batch', type=int, default=5, help='')
    parser.add_argument('--value_loss_coef', type=float, default=0.5, help='value loss coefficient (default: 0.5)')
    parser.add_argument('--entropy_coef', type=float, default=0.01, help='entropy term coefficient (default: 0.01)')
    parser.add_argument('--max_grad_norm', type=float, default=2.0, help='max norm of gradients (default: 0.5)')
    parser.add_argument('--no_entropy_decay', action='store_true',default=False,help='')

    # macro critic
    parser.add_argument('--macro_critic_epoch', type=int, default=10, help='number of ppo epochs (default: 4)')
    parser.add_argument('--macro_critic_batch', type=int, default=32, help='')
    parser.add_argument('--value_agg', type=str, default='sum', choices=['sum', 'mean'], help='aggregation for micro values in a macro grid')
    parser.add_argument('--loss_type', type=str, default='error', choices=['relative', 'sigmoid-relative','sigmoid-abs','error','abs-error','abs-relative'])
    parser.add_argument('--macro_level', type=str, default='od', choices=['od','o'], help='user od samples or o samples for macro critic optimization')
    parser.add_argument('--macro_coef', type=float, default=0.0, help='coefficient of the macro value')
    parser.add_argument('--macro_interval', type=int, default=5, help='')
    parser.add_argument('--macro_num', type=int, default=10000, help='')
    parser.add_argument('--hour_agg', type=int, default=6, help='')
    parser.add_argument('--no_macro',action='store_true',default=False,help='Disable macro training')
    parser.add_argument('--cooperative_interval',type=int,default=1,help='')
    parser.add_argument('--macro_num_layers',type=int,default=10,help='')
    


    # gravity
    parser.add_argument('--gravity_activation', type=str, default='relu', choices=['relu','tanh'])
    parser.add_argument('--gravity_loss', type=str, default='mse', choices=['mse','cross'])
    parser.add_argument('--gravity_softmax', type=int, default=0, choices=[0,1])
    
    # gail
    parser.add_argument('--gail_epoch', type=int, default=3, help='gail epochs (default: 5)')
    parser.add_argument('--gail_batch_size',type=int, default=2048,help='gail batch size (default: 128)')


    # pretrain
    parser.add_argument('--gravity_pretrain_epoch', type=int, default=200, help='gail epochs (default: 5)')
    parser.add_argument('--detach',action='store_true',default=False,help='')
    parser.add_argument('--macro_detach',action='store_true',default=False,help='')
    parser.add_argument('--low_first',action='store_true',default=False,help='')
    parser.add_argument('--gravity_batch', type=int, default=500, help='')
    parser.add_argument('--actor_high_pretrain_epoch', type=int, default=100, help='gail epochs (default: 5)')
    parser.add_argument('--actor_low_pretrain_epoch', type=int, default=100, help='gail epochs (default: 5)')
    parser.add_argument('--pretrain_id', type=int, default=100, help='')
    parser.add_argument('--with_pretrain_id', action='store_true',default=False)
    parser.add_argument('--with_pretrain_actor',action='store_true',default=False,help='enable pretrained actor')
    parser.add_argument('--pretrain_test',action='store_true',default=False,help='')
    parser.add_argument('--no_pretrain',action='store_true',default=False,help='disaable pretraining')

    parser.add_argument(
        '--gail-experts-dir',
        default='./gail_experts',
        help='directory that contains expert demonstrations for gail')

    
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.99,
        help='RMSprop optimizer apha (default: 0.99)')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='discount factor for rewards (default: 0.99)')
    parser.add_argument(
        '--use-gae',
        action='store_true',
        default=False,
        help='use generalized advantage estimation')
    parser.add_argument(
        '--gae-lambda',
        type=float,
        default=0.98,
        help='gae lambda parameter (default: 0.95)')
    
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    
    parser.add_argument(
        '--log-interval',
        type=int,
        default=10,
        help='log interval, one log per n updates (default: 10)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=100,
        help='save interval, one save per n updates (default: 100)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=None,
        help='eval interval, one eval per n updates (default: None)')
    parser.add_argument(
        '--num-env-steps',
        type=int,
        default=10e6,
        help='number of environment steps to train (default: 10e6)')
    parser.add_argument(
        '--env-name',
        default='PongNoFrameskip-v4',
        help='environment to train on (default: PongNoFrameskip-v4)')
    parser.add_argument(
        '--log-dir',
        default='/tmp/gym/',
        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument(
        '--save-dir',
        default='./trained_models/',
        help='directory to save agent logs (default: ./trained_models/)')
    parser.add_argument(
        '--no_cuda',
        action='store_true',
        default=False,
        help='Disable CUDA training')
    parser.add_argument(
        '--use-proper-time-limits',
        action='store_true',
        default=False,
        help='compute returns taking into account time limits')
    parser.add_argument(
        '--recurrent-policy',
        action='store_true',
        default=False,
        help='use a recurrent policy')
    parser.add_argument(
        '--use-linear-lr-decay',
        action='store_true',
        default=False,
        help='use a linear schedule on the learning rate')
    args = parser.parse_args()

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    return args
