import numpy as np

from algorithms.train_utils import *

def HAST(clients, args):
    
    num_clients = args.num_clients
    neighbours = np.arange(num_clients)

    for round in range(args.n_rounds):
        for i in range(num_clients):
            
            if clients[i].has_stopped:
                continue
            
            ## Random Sampling with Full Network Aggregation
            sampled_idx = np.random.choice(neighbours, args.n_sampled, 
                                        p=clients[i].random_priors, 
                                        replace=False)
            random_aggregation(sampled_idx, neighbours, clients, i)
        
            ## Simiarity Sampling with only Classifier Aggregation
            sampled_idx = np.random.choice(neighbours, args.n_sampled, 
                                           p=clients[i].priors_norm, 
                                           replace=False)
            similarity_aggregation(sampled_idx, clients, i)
            
        # Ensure parallel updates
        for i in range(num_clients):
            
            if clients[i].has_stopped:
                continue
            
            clients[i].priors = copy.deepcopy(clients[i].new_priors)
            clients[i].local_model = copy.deepcopy(clients[i].new_model) 
            
            not_i_idx = np.arange(len(clients)) != i
            clients[i].priors_norm[not_i_idx] = softmax_scale(
                np.clip(clients[i].priors[not_i_idx], 
                        a_min=c.DAC_MIN_SIMILARITY, 
                        a_max=c.DAC_MAX_SIMILARITY), args.tau)  
        
        ## Traditional model training
        train_clients(round, clients, args.local_epochs, finetune=args.HAST_finetune)
            
    return clients

def random_aggregation(sampled_idx:list[int], neighbours:list[int], clients:list, i:int) -> None:
    """Aggregates full model of the sampled client weights with local weights to create new model

    Args:
        sampled_idx (list[int]): Indexes of sampled clients
        neighbours (list[int]): Indexes of all available clients
        clients (_type_): List of all available clients
        i (int): Index of current client
    """
    model_weights, alpha, ik_loss = get_neighbour_info(clients, i, sampled_idx, 
                                                       use_new_model=False)
    
    ik_similarity = 1/ik_loss
    clients[i].n_sampled[sampled_idx] += 1
    clients[i].new_priors[sampled_idx] = ik_similarity
    
    w_new = hier_avg(model_weights, alpha, similarity_layers=False)            
    clients[i].new_model.load_state_dict(w_new)
    
    # Two hop
    new_neighbours = []
    base_set = set(neighbours[clients[i].n_sampled > 0])
    base_set.update([i])
    for k in sampled_idx:
        new_neighbours += set(neighbours[clients[k].priors > 0]) - base_set

    new_neighbours = np.unique(new_neighbours)
    
    for j in new_neighbours:
        best_prior, best_simil = 0, 0
        for info_idx, k in enumerate(sampled_idx):
            if(clients[k].priors[j] > 0 and ik_similarity[info_idx] > best_simil):
                best_simil = ik_similarity[info_idx]
                best_prior = clients[k].priors[j]

        clients[i].new_priors[j] = best_prior

def similarity_aggregation(sampled_idx:list[int], clients:list, i:int) -> None:
    """Aggregates a set number of layers from the sampled clients with the local client
       weights to get a new model

    Args:
        sampled_idx (list[int]): Indexes of sampled clients
        clients (list): List of all available clients
        i (int): Index of current client
    """

    clients[i].n_sampled[sampled_idx] += 1
    
    model_weights, alpha, _ = get_neighbour_info(clients, i, sampled_idx, 
                                                       use_new_model=True)
    
    w_new = hier_avg(model_weights, alpha, similarity_layers=True)            
    
    clients[i].new_model.load_state_dict(w_new)