import argparse
from argparse import Namespace
import algorithms.variables as v

from algorithms.Random import *
from algorithms.DAC import *
from algorithms.DAC2 import *
from algorithms.HAST import *


framework_functions = {
    "DAC"   : DAC,
    "DAC2"  : DAC2,
    "HAST" : HAST,
    "random": random
    }

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_time_steps', type=int, default=4, help='number of times we introduce new data to the clients')
    parser.add_argument('--n_rounds',   type=int,   default=100,    help="rounds of training")
    parser.add_argument('--num_clients',type=int,   default=100,    help="number of clients: K")
    parser.add_argument('--n_sampled',  type=int,   default=5,      help="number of clients to sample")
    
    parser.add_argument('--bs',         type=int,   default=8,      help="batch size")
    parser.add_argument('--lr',         type=float, default=0.01,   help="learning rate")
    parser.add_argument('--local_epochs',type=int,   default=3,      help="the number of local epochs: E")
    parser.add_argument('--n_data_train',type=int,  default=100,    help="train size")
    parser.add_argument('--n_data_val', type=int,   default=100,    help="validation size")
    parser.add_argument('--dataset',    type=str,   default='cifar10', help="name of dataset")
    parser.add_argument('--iid',        type=str,   default='covariate',help="covariate or label (type of shift)")
    parser.add_argument('--seed',       type=int,   default=1,      help='random seed (default: 1)')

    parser.add_argument('--tau',        type=int, default=1,    help="temperature in softmax")
    parser.add_argument('--num_classes',type=int,   default=10,     help="number of classes")
    parser.add_argument('--n_clusters', type=int,   default=4)    
    
    parser.add_argument('--framework',  type=str,   default='DAC',  help="Select framework for run, look in option to see availabel models.")
    parser.add_argument('--alt_tau',    action='store_true',        help="True selects the alternative tsu function")

    parser.add_argument('--oracle_intra_cluster_prob', type = int, default=1.0, help='Probability to communicate within your own cluster for oracle')

    parser.add_argument('--hierarchichal_aggregation_depth', type = int, default=3, help='Number of layers that should be randomly/similariy aggregated')
    parser.add_argument('--HAST_finetune', type = bool, default=True, help='Toggle if HAST should finetune similarity layers or not')
    parser.add_argument('--DAC_finetune', type = bool, default=False, help='Toggle if DAC should finetune similarity layers or not')
    parser.add_argument('--random_finetune', type = bool, default=False, help='Toggle if random should finetune similarity layers or not')
    parser.add_argument('--cluster_function', type=str, default='randomize_clusters', help='select cluster function for each timestep')
    parser.add_argument('--conditional_probabilities', type=bool, default=False, help='Toggle to enable label swaps between time steps')
    
    parser.add_argument('--always_new_data_at_timestep', type=bool, default=True, help='Toggle to enable clients that remain in the same cluster to get new data at time steps')
    parser.add_argument('--early_stopping_aggregation', type=bool, default=False, help='toggle if we require validation loss improvement to update model with (fedavg) aggregate')
    

    args = parser.parse_args()
    return args

# ---------------- Ready made experiments for easier reference -----------------------

# Use parameters from terminal input
default = {}

### Dataset Base Parameters ###
###############################
cifar10_params = {
    "n_rounds" : 200,
    "num_time_steps": 10,
    "bs" : 8,
    "local_epochs" : 3,
    "n_data_train" : 250,
    "n_data_val": 75,
    "dataset": "cifar10",
    "num_classes": 10,
}

pacs_2_params = {
    "n_rounds" : 200,
    "num_time_steps": 4,
    "bs" : 8,
    "local_epochs" : 3,
    "n_data_train" : 175,
    "n_data_val": 75,
    "dataset": "pacs",
    "num_classes": 7,
    "metadata" : {'dataset' : ['pacs0', 'pacs1']}
}

pacs_4_params = {
    "n_rounds" : 200,
    "num_time_steps": 4,
    "bs" : 8,
    "local_epochs" : 3,
    "n_data_train" : 175,
    "n_data_val": 75,
    "dataset": "pacs",
    "num_classes": 7,
    "metadata" : {'dataset' : ['pacs0', 'pacs1', 'pacs2', 'pacs3']}
}


### Algorithm Base Parameters ###
#################################
random_params = {
    "framework": 'random',
    "n_sampled": 6,
    "random_finetune": False,
}

DAC_params = {
    "framework": 'DAC',
    "n_sampled": 6,
    "tau": 30,
    "DAC_finetune": False
}

HAST_params = {
    "framework": 'HAST',
    "n_sampled": 3,
    "tau": 30,
    "hierarchichal_aggregation_depth": 3,
    "hierarchichal_random_aggregation_all_layers": True,
    "HAST_finetune": True,
}

### Pattern Base Parameters ###
###############################
feddrift_pattern_2_cluster_params = {
    "cluster_function": "fed_drift_2_cluster_v2",
    "num_clients" : 20,
    "n_clusters": 2,
    "cluster_sizes" : [1, 0],
    "always_new_data_at_timestep": False,
}

feddrift_pattern_4_cluster_params = {
    "cluster_function": "fed_drift_4_cluster_v2",
    "num_clients" : 20,
    "n_clusters": 4,
    "cluster_sizes" : [1, 0, 0, 0],
    "always_new_data_at_timestep": False,
}

random_pattern_1_cluster_params = {
    "cluster_function": "randomize_clusters",
    "num_clients" : 50,
    "n_clusters": 1,
    "cluster_sizes" : [1, 0],
    "always_new_data_at_timestep": True,
}

random_pattern_2_cluster_params = {
    "cluster_function": "randomize_clusters",
    "num_clients" : 50,
    "n_clusters": 2,
    "cluster_sizes" : [0.5, 0.5],
    "always_new_data_at_timestep": True,
}

random_pattern_4_cluster_params = {
    "cluster_function": "randomize_clusters",
    "num_clients" : 50,
    "n_clusters": 4,
    "cluster_sizes" : [0.25, 0.25, 0.25, 0.25],
    "always_new_data_at_timestep": True,
}


### Experiment Parameters ###
#############################
label_swap_2_cluster = {
    "iid": "label",
    "label_groups_cifar10" : [[0,1,3,4,5,7,8,9],[0,1,3,4,5,7,8,9]],
    "label_groups_mnist" : [[0,1,3,4,5,7,8,9],[0,1,3,4,5,7,8,9]],
    "label_groups_pacs" : [[0,1,2,3,4,5,6],[0,1,2,3,4,5,6]],
    "conditional_probabilities" : True,
    "conditional_numbs": [[], [0,1]],
    "random_lr": 0.00008,
    "DAC_lr": 0.00008,
    "HAST_lr": 0.00008, 
}

label_swap_4_cluster = {
    "iid": "label",
    "label_groups_cifar10" : [[0,1,3,4,5,7,8,9],[0,1,3,4,5,7,8,9],[0,1,3,4,5,7,8,9],[0,1,3,4,5,7,8,9]],
    "label_groups_mnist" : [[0,1,3,4,5,7,8,9],[0,1,3,4,5,7,8,9],[0,1,3,4,5,7,8,9],[0,1,3,4,5,7,8,9]],
    "conditional_probabilities" : True,
    "conditional_numbs": [[], [0,1], [3,4], [5,7]],
    "random_lr": 0.00008,
    "DAC_lr": 0.00008,
    "HAST_lr": 0.00008, 
}

label_drift_1_cluster = {
    "iid": "label",
    "label_groups_cifar10" : [[0,1,2,3,4,5,6,7,8,9]],
    "label_groups_mnist" : [[0,1,2,3,4,5,6,7,8,9]],
    "label_groups_pacs" : [[0,1,2,3,4,5,6],[0,1,2,3,4,5,6]],
    "random_lr": 0.00008,
    "DAC_lr": 0.00008,
    "HAST_lr": 0.00008, 
}

label_drift_2_cluster = {
    "iid": "label",
    "label_groups_cifar10" : [[0,1,8,9],[3,4,5,7]],
    "label_groups_mnist" : [[0,1,8,9],[3,4,5,7]],
    "label_groups_pacs" : [[0,1,2,3,4,5,6],[0,1,2,3,4,5,6]],
    "random_lr": 0.00008,
    "DAC_lr": 0.00008,
    "HAST_lr": 0.00008, 
}

label_drift_4_cluster = {
    "iid": "label",
    "label_groups_cifar10" : [[0,1],[3,4],[5,7],[8,9]],
    "label_groups_mnist" : [[0,1],[3,4],[5,7],[8,9]],
    "label_groups_pacs" : [[0,1,2,3,4,5,6],[0,1,2,3,4,5,6],[0,1,2,3,4,5,6],[0,1,2,3,4,5,6]],
    "random_lr": 0.00008,
    "DAC_lr": 0.00008,
    "HAST_lr": 0.00008, 
}



## FedDrift Cluster Function
cifar10_2_cluster_label_swap_feddrift = {
    "base_dataset" : 'cifar10_params',
    "drift_pattern": 'feddrift_pattern_2_cluster_params',
    "experiment_params": 'label_swap_2_cluster',
}

cifar10_4_cluster_label_swap_feddrift = {
    "base_dataset" : 'cifar10_params',
    "drift_pattern": 'feddrift_pattern_4_cluster_params',
    "experiment_params": 'label_swap_4_cluster',
}

cifar10_2_cluster_label_drift_feddrift = {
    "base_dataset" : 'cifar10_params',
    "drift_pattern": 'feddrift_pattern_2_cluster_params',
    "experiment_params": 'label_drift_2_cluster',
}

cifar10_4_cluster_label_drift_feddrift = {
    "base_dataset" : 'cifar10_params',
    "drift_pattern": 'feddrift_pattern_4_cluster_params',
    "experiment_params": 'label_drift_4_cluster',
}

pacs_2_cluster_label_swap_feddrift = {
    "base_dataset" : 'pacs_2_params',
    "drift_pattern": 'feddrift_pattern_2_cluster_params',
    "experiment_params": 'label_swap_2_cluster',
}

pacs_4_cluster_label_swap_feddrift = {
    "base_dataset" : 'pacs_4_params',
    "drift_pattern": 'feddrift_pattern_4_cluster_params',
    "experiment_params": 'label_swap_4_cluster',
}

pacs_2_cluster_label_drift_feddrift = {
    "base_dataset" : 'pacs_2_params',
    "drift_pattern": 'feddrift_pattern_2_cluster_params',
    "experiment_params": 'label_drift_2_cluster',
}

pacs_4_cluster_label_drift_feddrift = {
    "base_dataset" : 'pacs_4_params',
    "drift_pattern": 'feddrift_pattern_4_cluster_params',
    "experiment_params": 'label_drift_4_cluster',
}

## Random Cluster Function
cifar10_2_cluster_label_swap_random = {
    "base_dataset" : 'cifar10_params',
    "drift_pattern": 'random_pattern_2_cluster_params',
    "experiment_params": 'label_swap_2_cluster',
}

cifar10_4_cluster_label_swap_random = {
    "base_dataset" : 'cifar10_params',
    "drift_pattern": 'random_pattern_4_cluster_params',
    "experiment_params": 'label_swap_4_cluster',
}

cifar10_1_cluster_label_drift_random = {
    "base_dataset" : 'cifar10_params',
    "drift_pattern": 'random_pattern_1_cluster_params',
    "experiment_params": 'label_drift_1_cluster',
}

cifar10_2_cluster_label_drift_random = {
    "base_dataset" : 'cifar10_params',
    "drift_pattern": 'random_pattern_2_cluster_params',
    "experiment_params": 'label_drift_2_cluster',
}

cifar10_4_cluster_label_drift_random = {
    "base_dataset" : 'cifar10_params',
    "drift_pattern": 'random_pattern_4_cluster_params',
    "experiment_params": 'label_drift_4_cluster',
}

pacs_2_cluster_label_swap_random = {
    "base_dataset" : 'pacs_2_params',
    "drift_pattern": 'random_pattern_2_cluster_params',
    "experiment_params": 'label_swap_2_cluster',
}

pacs_4_cluster_label_swap_random = {
    "base_dataset" : 'pacs_4_params',
    "drift_pattern": 'random_pattern_4_cluster_params',
    "experiment_params": 'label_swap_4_cluster',
}

pacs_2_cluster_label_drift_random = {
    "base_dataset" : 'pacs_2_params',
    "drift_pattern": 'random_pattern_2_cluster_params',
    "experiment_params": 'label_drift_2_cluster',
}

pacs_4_cluster_label_drift_random = {
    "base_dataset" : 'pacs_4_params',
    "drift_pattern": 'random_pattern_4_cluster_params',
    "experiment_params": 'label_drift_4_cluster',
}


experiment_params = {
    "default": default,
    
    # Datasets
    "cifar10_params": cifar10_params,
    "pacs_2_params": pacs_2_params,
    "pacs_4_params": pacs_4_params,
    
    # Algorithms
    "random_params": random_params,
    "DAC_params": DAC_params,
    "HAST_params": HAST_params,
    
    # Drift Patterns
    "feddrift_pattern_2_cluster_params": feddrift_pattern_2_cluster_params,
    "feddrift_pattern_4_cluster_params": feddrift_pattern_4_cluster_params,
    "random_pattern_1_cluster_params": random_pattern_1_cluster_params,
    "random_pattern_2_cluster_params": random_pattern_2_cluster_params,
    "random_pattern_4_cluster_params": random_pattern_4_cluster_params,
    
    # Experiments    
    "label_swap_2_cluster": label_swap_2_cluster,
    "label_swap_4_cluster": label_swap_4_cluster,
    "label_drift_1_cluster": label_drift_1_cluster,
    "label_drift_2_cluster": label_drift_2_cluster,
    "label_drift_4_cluster": label_drift_4_cluster,



    ## Experiments
    "cifar10_2_cluster_label_swap_feddrift": cifar10_2_cluster_label_swap_feddrift,
    "cifar10_4_cluster_label_swap_feddrift": cifar10_4_cluster_label_swap_feddrift,
    "cifar10_2_cluster_label_drift_feddrift": cifar10_2_cluster_label_drift_feddrift,
    "cifar10_4_cluster_label_drift_feddrift": cifar10_4_cluster_label_drift_feddrift,
    
    "pacs_2_cluster_label_swap_feddrift": pacs_2_cluster_label_swap_feddrift,
    "pacs_4_cluster_label_swap_feddrift": pacs_4_cluster_label_swap_feddrift,
    "pacs_2_cluster_label_drift_feddrift": pacs_2_cluster_label_drift_feddrift,
    "pacs_4_cluster_label_drift_feddrift": pacs_4_cluster_label_drift_feddrift,
    
    "cifar10_2_cluster_label_swap_random": cifar10_2_cluster_label_swap_random,
    "cifar10_4_cluster_label_swap_random": cifar10_4_cluster_label_swap_random,
    "cifar10_2_cluster_label_drift_random": cifar10_2_cluster_label_drift_random,
    "cifar10_4_cluster_label_drift_random": cifar10_4_cluster_label_drift_random,
    
    "pacs_2_cluster_label_swap_random": pacs_2_cluster_label_swap_random,
    "pacs_4_cluster_label_swap_random": pacs_4_cluster_label_swap_random,
    "pacs_2_cluster_label_drift_random": pacs_2_cluster_label_drift_random,
    "pacs_4_cluster_label_drift_random": pacs_4_cluster_label_drift_random,
    
    "cifar10_1_cluster_label_drift_random": cifar10_1_cluster_label_drift_random,
}

#---------------------------------------------------------------
#                 HELPER FUNCTIONS RUN_EXPERIMENT
#---------------------------------------------------------------

def check_args(args):
          
    valid_datasets = ['synthetic', 'cifar10', 'mnist', 'fashion-mnist', 'svhn', 'cifar100', 'pacs']
    valid_iids = ['covariate', 'label']
    safety_arguments = {
                        'num_time_steps' :          1,
                        'n_rounds' :                1,
                        'num_clients' :             1,
                        'n_sampled':                1, 
                        'bs' :                      1,
                        'lr':                       0,
                        'local_epochs' :            1,
                        'n_data_train' :            1, 
                        'n_data_val':               1, 
                        'seed':                     0,
                        'tau':                      1,
                        'n_clusters':               1,
                        }    
    
    args = vars(args)
    error_count = 1
    print('Checking experiment arguments...')
    
    error_text = ('-'*45 + '\n')    
    
    if args['dataset'] == 'synthetic':
        if 'means' not in args:
            error_text += (f'{error_count}:\tMissing argument: means\n')
            error_count += 1
        elif len(args['means']) != args['n_clusters']:
            error_text += ('{}:\tNumber of means ({}) does not match n_clusters ({})\n'.format(error_count, len(args['means']), args['n_clusters']))
            error_count += 1 
    else:
        if args['iid'] == 'covariate': 
            if 'rotations' not in args:
                error_text += (f'{error_count}:\tMissing argument: rotations\n')
                error_count += 1
            elif len(args['rotations']) != args['n_clusters']:
                error_text += ('{}:\tNumber of rotations ({}) does not match n_clusters ({})\n'.format(error_count, len(args['rotations']), args['n_clusters']))
                error_count += 1
        if args['iid'] == 'label': 
            if 'label_groups' not in args:
                error_text += (f'{error_count}:\tMissing argument: label_groups\n')
                error_count += 1
            elif len(args['label_groups']) != args['n_clusters']:
                error_text += ('{}:\tNumber of label_groups ({}) does not match n_clusters ({})\n'.format(error_count, len(args['rotations']), args['n_clusters']))
                error_count += 1   
          
    for arg in safety_arguments:
        if type(args[arg]) != type(safety_arguments[arg]):
            if arg == 'lr' and type(args[arg]) == float:
                continue
            else:
                error_text += (f'{error_count}:\t{arg} has wrong input type {type(args[arg])} instead of {type(safety_arguments[arg])}\n')
                error_count += 1
        else:
            if args[arg] < safety_arguments[arg]:
                error_text += (f'{error_count}:\t{arg} value is below the threshold of {safety_arguments[arg]}\n')
                error_count += 1
    
    if args['dataset'] not in valid_datasets:
        error_text += ('{}:\t{} is not implemented! Try one of {}\n'.format(error_count, args['dataset'], valid_datasets))
        error_count += 1
    if args['iid'] not in valid_iids:
        error_text += ('{}:\t{} is not implemented! Try one of {}\n'.format(error_count, args['iid'], valid_iids))
        error_count += 1
    
    
    if 'cluster_sizes' not in args:
        error_text += (f'{error_count}:\tMissing argument: cluster_sizes\n')
        error_count += 1
    else:
        if len(args['cluster_sizes']) != args['n_clusters']:
            error_text += ('{}:\tNumber of cluster_sizes ({}) does not match n_clusters ({})\n'.format(error_count, len(args['cluster_sizes']), args['n_clusters']))
            error_count += 1
        if sum(args['cluster_sizes'])  < 0.95:
            error_text += ('{}:\tSizes of cluster_sizes ({}) is not sufficiently close to (1)\n'.format(error_count, sum(args['cluster_sizes'])))
            error_count += 1
    
    args = Namespace(**args)
    
    if error_count != 1:
        error_text += ('-'*45 + f'\nError: User arguments are not compatible, please see the above {error_count-1} issue(s)!\n')
        print(error_text)
        exit()
    
def define_global_vars(args):
    
    v.datasets = {}
    v.transforms = {}
    v.criterions = None
    v.clients_per_cluster = []
    
    v.label_ids = {}
    v.label_ids_used = {}
    
    v.warning_message = ''
    
    v.experiment = args.iid
    
    if hasattr(args, 'rotations'): v.rotations_list = args.rotations
        
    if hasattr(args, 'means'):
        v.means_list = args.means
        
    if hasattr(args, 'label_groups'):
        v.label_groups = args.label_groups
    
    if hasattr(args, 'metadata'):
        v.metadata = args.metadata
    else:
        v.metadata =  {}
        v.metadata['num_channels'] = None
        v.metadata['img_size'] = None       
        
    v.hierarchichal_aggregation_depth = args.hierarchichal_aggregation_depth * 2
    v.hierarchichal_random_aggregation_all_layers = args.hierarchichal_random_aggregation_all_layers
    v.conditional_probabilities = args.conditional_probabilities