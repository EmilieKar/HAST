import copy
import numpy as np
import algorithms.constants as c
import algorithms.variables as v
import algorithms.local_models as models
from torch import add

def initialize_priors_and_models(clients, args):
    num_clients = len(clients)
    for i in range(num_clients):
        clients[i].train(args.local_epochs)
        priors = np.ones(num_clients)           # initial uniform probability
        priors[i] = 0.0                         # probability to sample yourself is 0
        prior = priors/np.sum(priors)
        clients[i].priors = prior
        clients[i].priors_norm = copy.deepcopy(prior)
    
    if args.framework == 'HAST':
        for client in clients:
            client.random_priors = copy.deepcopy(client.priors)
            
    test_clients(clients)

def initialize_models(clients, args):
    for client in clients:
        client.train(args.local_epochs, test=True)    

def train_clients(round, clients, n_local_epochs, finetune=False):
    num_clients = len(clients)
    accs = np.zeros(num_clients)
    train_losses = np.zeros(num_clients)
    val_losses = np.zeros(num_clients)
    
    for i in range(num_clients): 
        _, train_loss, val_loss, val_acc = clients[i].train(n_local_epochs, finetune=finetune)
        accs[i] = np.round(val_acc,2)
        train_losses[i] = np.round(train_loss, 2)
        val_losses[i] = np.round(val_loss, 2)
    
    if round % c.NUM_ROUNDS_BETWEEN_TESTS == 0:
        test_clients(clients)
    else:
        copy_test_scores(clients)
        
    check_early_stopping(clients)

    print(f"Round {round} | Avg train loss {np.round(np.mean(train_losses),2)} | Avg val loss {np.round(np.mean(val_losses),2)} | Avg val acc {np.round(np.mean(accs),2)}")  

def update_best_models(clients):
    """Resets best model values at new time steps"""
    
    for client in clients:
        val_loss, val_acc = client.validate(client.best_model, train_set = False)
        client.best_val_loss = val_loss
        client.best_val_acc = val_acc
        client.patience_counter = 0

def check_early_stopping(clients):
    for client in clients:
        if client.patience_counter > c.EARLY_STOPPING_PATIENCE:
            client.early_stopped[client.current_round] = 1
            client.has_stopped = True

def test_clients(clients):
    round = clients[0].current_round
    
    # Validate model on completely unseen/unvalidated test data
    for client in clients:
        
        # If has_stopped, previous values will already be stored.
        if client.has_stopped:
            continue
        
        cluster_idx = client.cluster_id[-1]
        
        if client.dataset == 'pacs':
            dataset_name = f'{client.dataset}{cluster_idx}'
        elif v.conditional_probabilities:
            dataset_name = v.dataset_names[cluster_idx + 1]
        else:
            dataset_name = client.dataset
            
        test_loader = v.testdata[dataset_name][cluster_idx]
        
        test_val_loss, test_val_acc = client.validate(client.best_model, 
                                                    dataloader=test_loader)
        client.best_test_loss_list[round] = test_val_loss
        client.best_test_acc_list[round] = test_val_acc
            
def copy_test_scores(clients):
    round = clients[0].current_round
    
    for client in clients:
        client.best_test_loss_list[round] = client.best_test_loss_list[round - 1]
        client.best_test_acc_list[round] = client.best_test_acc_list[round - 1]

def get_neighbour_info(clients, client_idx, sampled_client_idxs, use_new_model=False):
    model_weights = np.empty(len(sampled_client_idxs) + 1, dtype=models.CNN)
    nmb_datapoints = np.zeros(len(sampled_client_idxs) + 1, dtype=int)
    ik_loss = np.zeros(len(sampled_client_idxs), dtype=float)
        
    for i, k in enumerate(sampled_client_idxs):

        model_weights[i+1] = clients[k].local_model.state_dict()
        nmb_datapoints[i+1] = len(clients[k].ldr_train.dataset)
        
        ik_loss[i], _ = clients[client_idx].validate(clients[k].local_model, train_set=True)
    
    ik_loss = np.clip(ik_loss, a_min=c.MIN_LOSS, a_max=None)
    
    if use_new_model:
        model_weights[0] = clients[client_idx].new_model.state_dict()
    else:
        model_weights[0] = clients[client_idx].local_model.state_dict()
    nmb_datapoints[0] = len(clients[client_idx].ldr_train.dataset)

    return model_weights, nmb_datapoints, ik_loss

def hier_avg(models, alpha, similarity_layers=False):
    new_model_weights = copy.deepcopy(models[0])
    alpha_list = alpha/np.sum(alpha)

    all_keys = new_model_weights.keys()
    if similarity_layers:                                                       # Similarity Aggregation
        if v.hierarchichal_aggregation_depth == 0:
            return new_model_weights
        keys = list(all_keys)[-v.hierarchichal_aggregation_depth:]
    else:                                                                       # Random Aggregation
        if v.hierarchichal_random_aggregation_all_layers:
            keys = all_keys
        else:
            if v.hierarchichal_aggregation_depth == 0:
                keys = all_keys
            else:
                keys = list(all_keys)[0:-v.hierarchichal_aggregation_depth]
            
    for layer in keys: 
        new_model_weights[layer].zero_()
        for i, model in enumerate(models): 
            # out = input + alpha x other (~20% faster than old method)
            add(input=new_model_weights[layer], other=model[layer], 
                alpha=alpha_list[i], out=new_model_weights[layer])
            
    return new_model_weights

def fed_avg(models,nmb_datapoints):
    new_model_weights = copy.deepcopy(models[0])
    alpha_list = nmb_datapoints/np.sum(nmb_datapoints)

    for layer in new_model_weights.keys(): 
        new_model_weights[layer].zero_()

        for i, model in enumerate(models): 
            add(input=new_model_weights[layer], other=model[layer], 
                alpha=alpha_list[i], out=new_model_weights[layer])
            
    return new_model_weights
    

def weighted_avg(models, similarities):
    new_model_weights = copy.deepcopy(models[0])
    client_alpha = 1/(len(models)) 

    alpha_list = similarities / np.sum(similarities)
    alpha_list = alpha_list * (1 - client_alpha)
    alpha_list = np.insert(alpha_list, 0, client_alpha)

    for layer in new_model_weights.keys(): 
        new_model_weights[layer].zero_()

        for i, model in enumerate(models): 
            add(input=new_model_weights[layer], other=model[layer], 
                alpha=alpha_list[i], out=new_model_weights[layer])
            
    return new_model_weights

def softmax_scale(x, tau):
    x_new = np.exp(x*tau)/sum(np.exp(x*tau))
    return x_new

def tau_function(x,a,b):
    tau = 2*a/(1+np.exp(-b*x)) - a +1
    return tau