import numpy as np
import random as rand
import git
import pickle
import json
import os
import datetime

from algorithms.client import *
from data_utils import *
from algorithms.train_utils import *
from utilities import *
from utilities import args_parser, experiment_params, framework_functions

def reinitialize_models(clients, model):
    for client in clients:
        client.reinitialize_models(model)

def set_best_model_as_local_model(clients):
    for client in clients:
        client.local_model = copy.deepcopy(client.best_model)
        
def get_initialized_model(dataset, data, device):
    
    input_dim = data[0][0].size()[0]
    image_size = data[0][0].size()[1]
        
    if image_size == 32: size_dim = 25
    elif image_size == 28: size_dim = 16
    elif image_size == 227: size_dim = 25
    elif image_size != 2:
        raise Exception(f'Unknown image size detected: {image_size}')
    
    
    if(dataset=='cifar10'):
        local_model = CNN(input_dim=input_dim, 
                        num_classes=c.CIFAR10_NUM_CLASSES, 
                        size_dim=size_dim).to(device)
    elif (dataset=='pacs'):
        local_model = CNN_PACS(input_dim=input_dim, 
                        num_classes=c.PACS_NUM_CLASSES, 
                        size_dim=size_dim).to(device)
    
    return local_model

# --------------------------------------------------------------------------------------------------------
#                                           CLUSTER FUNCTIONS
# --------------------------------------------------------------------------------------------------------
def rotate_clusters(clients, args):
        
    for client in clients:
        client.cluster_id.append((client.cluster_id[-1] + 1) % args.n_clusters)

def randomize_clusters(clients, args):
        
    cluster_id_list = [client.cluster_id[-1] for client in clients]
    rand.shuffle(cluster_id_list)

    for i in range(args.num_clients):
        clients[i].cluster_id.append(cluster_id_list[i])

def maintain_clusters(clients, args):
    for client in clients:
        client.cluster_id.append(client.cluster_id[-1])

def fed_drift_2_cluster(clients, timestep):

    cluster_pattern = [
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [1,1,0,0,0,0,0,0,0,0],
        [1,1,1,1,1,0,0,0,0,0],
        [1,1,1,1,1,1,0,0,0,0],
        [1,1,1,1,1,1,1,1,0,0],
        [1,1,1,1,1,1,1,1,0,0],
        [1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1]
    ]
    timestep_row = cluster_pattern[timestep]

    for idx in range(10):
        clients[idx].cluster_id.append(timestep_row[idx])

def fed_drift_2_cluster_v2(clients, timestep):

    cluster_pattern = [
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0],
        [1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0],
        [1,1,1,1,1,1,0,0,0,0,1,1,1,1,1,1,0,0,0,0],
        [1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,0,0],
        [1,1,1,1,1,1,1,1,0,0,1,1,1,1,1,1,1,1,0,0],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    ]
    timestep_row = cluster_pattern[timestep]

    for idx in range(20):
        clients[idx].cluster_id.append(timestep_row[idx])

def fed_drift_4_cluster(clients, timestep):

    cluster_pattern = [
        [0,0,0,0,0,0,0,0,0,0],
        [1,1,1,2,2,2,0,0,0,0],
        [1,1,1,2,2,2,0,0,3,0],
        [2,2,1,1,2,2,2,1,3,0],
        [2,2,2,1,2,3,2,1,3,0],
        [2,3,2,1,1,3,3,1,3,3],
        [3,3,2,3,1,3,3,2,1,3],
        [3,0,3,3,3,1,3,2,1,3],
        [0,0,3,3,3,1,2,2,2,3],
    ]
    timestep_row = cluster_pattern[timestep]

    for idx in range(10):
        clients[idx].cluster_id.append(timestep_row[idx])
        
def fed_drift_4_cluster_v2(clients, timestep):

    cluster_pattern = [
        [0,0,0,0,0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0],
        [1,1,1,2,2,2,0,0,0,0, 1,1,1,2,2,2,0,0,0,0],
        [1,1,1,2,2,2,0,0,3,0, 1,1,1,2,2,2,0,0,3,0],
        [2,2,1,1,2,2,2,1,3,0, 2,2,1,1,2,2,2,1,3,0],
        [2,2,2,1,2,3,2,1,3,0, 2,2,2,1,2,3,2,1,3,0],
        [2,3,2,1,1,3,3,1,3,3, 2,3,2,1,1,3,3,1,3,3],
        [3,3,2,3,1,3,3,2,1,3, 3,3,2,3,1,3,3,2,1,3],
        [3,0,3,3,3,1,3,2,1,3, 3,0,3,3,3,1,3,2,1,3],
        [0,0,3,3,3,1,2,2,2,3, 0,0,3,3,3,1,2,2,2,3],
    ]
    timestep_row = cluster_pattern[timestep]

    for idx in range(20):
        clients[idx].cluster_id.append(timestep_row[idx])

cluster_functions = {
    "rotate_clusters" : rotate_clusters,
    "randomize_clusters" : randomize_clusters, 
    "maintain_clusters" : maintain_clusters,
    'fed_drift_2_cluster': fed_drift_2_cluster,
    'fed_drift_2_cluster_v2': fed_drift_2_cluster_v2,
    'fed_drift_4_cluster': fed_drift_4_cluster,
    'fed_drift_4_cluster_v2': fed_drift_4_cluster_v2
}

# --------------------------------------------------------------------------------------------------------
#                                            RUN EXPERIMENTS
# --------------------------------------------------------------------------------------------------------

def run_experiment(args):
    ## Initialization / Set Up
    check_args(args)
    define_global_vars(args)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    num_clients = args.num_clients

    torch.manual_seed(args.seed)
    rand.seed(args.seed)
    np.random.seed(args.seed)

    client_trainset, client_valset, cluster_ids, criterion = init_dataset(args)
    
    default_model = get_initialized_model(args.dataset, client_trainset[0], device)
    v.default_model = default_model
    
    clients = []
    for idx in range(num_clients):

        kwargs = {
            'lr':           args.lr,
            'batch_size':   args.bs,
            'num_clients':  num_clients,
            'criterion':    criterion,
            'device':       device,
            'dataset':      args.dataset,
            'cluster_id':   cluster_ids[idx],
            'total_rounds': args.num_time_steps * args.n_rounds,
            'num_clusters': args.n_clusters
        }                

        client = Client(train_data=client_trainset[idx], val_data=client_valset[idx], **kwargs)
        client.initialize_model(default_model)
        clients.append(client)

    initialize_priors_and_models(clients, args)

    for time_step in range(args.num_time_steps-1):
        print(f'Starting timestep {time_step}:')
        framework_function = framework_functions[args.framework]
        clients = framework_function(clients, args)
        
        if ('fed_drift' in args.cluster_function):
            cluster_functions[args.cluster_function](clients, time_step)
        else:
            cluster_functions[args.cluster_function](clients, args)

        if args.reinitialize_models:
            reinitialize_models(clients, default_model)
        elif args.set_best_model_as_local_model:
            set_best_model_as_local_model(clients)
        
        for client in clients:
            if args.always_new_data_at_timestep:
                new_train_dataset = generate_new_data(args.n_data_train, client.cluster_id[-1], time_step=time_step)
                new_val_dataset = generate_new_data(args.n_data_val, client.cluster_id[-1], time_step=time_step)
                client.create_data_loaders(new_train_dataset, new_val_dataset)
            else:
                if client.cluster_id[-1] != client.cluster_id[-2]:                  # Only update data if option = True, and we switchted cluster
                    new_train_dataset = generate_new_data(args.n_data_train, client.cluster_id[-1], time_step=time_step)
                    new_val_dataset = generate_new_data(args.n_data_val, client.cluster_id[-1], time_step=time_step)
                    client.create_data_loaders(new_train_dataset, new_val_dataset)

        update_best_models(clients)
        
    print(f'Starting timestep {args.num_time_steps -1 }:')
    framework_function = framework_functions[args.framework]
    clients = framework_function(clients, args)

    print('Experiment finished!')   
    return clients


def run_multiple_experiments(args, experiments, frameworks, seeds, var = None, optional_save_name = None):
    for experiment in experiments:
        for framework in frameworks:
            args_dict = vars(args)
            if var is not None:
                temp_variable = args_dict[var]
            args_dict.update(experiment_params[experiment])
            
            params = experiment_params[experiment]
            args_dict.update(experiment_params[params['base_dataset']])                 # Load dataset specific settings
            args_dict.update(experiment_params[f'{framework}_params'])                  # Load algorithm specific settings
            args_dict.update(experiment_params[params['experiment_params']])            # Load experiment specific settings
            args_dict.update(experiment_params[args_dict['drift_pattern']])             # Load pattern specific settings
            
            args_dict['lr'] = experiment_params[params['experiment_params']][f'{framework}_lr'] # Load lr for specific experiment and algorithm
            
            if args_dict['iid']:                                                        # Load correct label groups
                args_dict['label_groups'] = args_dict[f'label_groups_{args.dataset}']
            
            if var is not None:
                args_dict[var] = temp_variable
            
            for seed in seeds:
                args.seed = seed
                print(f'Starting experiment {experiment} with framework {framework} and random seed {seed}')

                clients = run_experiment(args)

                if optional_save_name is not None:
                    experiment_name = optional_save_name
                else:
                    experiment_name = experiment
                if var is not None:
                    exp_name = f'{experiment_name}_{framework}_{seed}_{var}_{args_dict[var]}'
                else:
                    exp_name = f'{experiment_name}_{framework}_{seed}'
                

                store_clients(clients, args, exp_name)
                                        
# --------------------------------------------------------------------------------------------------------
#                                        SAVING RUN INFO
# --------------------------------------------------------------------------------------------------------

def store_clients(clients, args, exp_name):
    print('Storing clients to disk')
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha

    save_path = os.getcwd() + '/save/runs/_unanalyzed/' + exp_name + '/'
    exists = os.path.exists(save_path)

    if not exists:
        os.makedirs(save_path)
        print(f'New run directory is created for experiment {exp_name}')

    idx = 0
    for client in clients: 
        # Drop unecessary info before storing
        client.ldr_train = None 
        client.ldr_val = None
        client.new_model = None
        client.received_models = []
        client.new_priors = None
        client.new_n_sampled = None

        pickle_path = save_path + f'client_{idx}.pkl'
        with open(pickle_path, "wb") as filehandler:
            pickle.dump(client, filehandler)

        idx += 1

    with open(save_path + '_config.txt', 'w') as filehandler:
        args_to_file = copy.copy(args)
        args_to_file.sha = sha
        json.dump(args_to_file.__dict__, filehandler, indent=2)
    
    if v.warning_message != '':
        with open(save_path + '_warnings.txt', 'w') as filehandler:
            filehandler.write(v.warning_message)
        
    
    with open(save_path + '_experiment_time.txt', 'w') as filehandler:
        filehandler.write(str(datetime.datetime.now()))

    print('Finished storing clients!')


# --------------------------------------------------------------------------------------------------------
#                                               MAIN
# --------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    args = args_parser()
    args_dict = vars(args)

    experiments = ['cifar10_2_cluster_label_drift_random']
    frameworks = ['HAST']
    seeds = [0]
    optional_save_name = None

    run_multiple_experiments(args, experiments, frameworks, seeds, var = None, optional_save_name=optional_save_name)
