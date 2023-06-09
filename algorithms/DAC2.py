import numpy as np

from algorithms.train_utils import *
import algorithms.constants as c

# This is an updated version of the DAC model with a sligthly altered two hop method as well as a similarity 
# based aggregation function. We didn't include this in our paper as HAST performed better, however this model 
# still performs better than the originally proposed DAC model

def DAC2(clients, args):

    num_clients = len(clients)
    neighbour_list = np.arange(num_clients)
    
    for round in range(args.n_rounds):
        for i in range(num_clients):
            
            if clients[i].has_stopped:
                continue
                
            sampled_client_idxs = np.random.choice(neighbour_list, args.n_sampled, replace=False, p=clients[i].priors_norm)

            model_weights, _, ik_loss = get_neighbour_info(clients, i, sampled_client_idxs)
            
            ik_similarity = [1/loss for loss in ik_loss]

            for k in sampled_client_idxs: 
                clients[i].new_n_sampled[k] += 1

            for k in range(args.n_sampled):
                clients[i].new_priors[sampled_client_idxs[k]] = ik_similarity[k]

            w_new = weighted_avg(model_weights, ik_similarity)
            clients[i].new_model.load_state_dict(w_new)

            new_neighbours = []
            for k in sampled_client_idxs:
                new_neighbours += list(set(neighbour_list[clients[k].priors > 0]) - set(
                    neighbour_list[clients[i].n_sampled > 0]) - set([i]))

            new_neighbours = np.unique(new_neighbours)

            for j in new_neighbours:
                ki_scores = []

                for info_idx, k in enumerate(sampled_client_idxs):
                    if(clients[k].priors[j] > 0):
                        ki_scores.append(
                            (clients[k].priors[j], ik_similarity[info_idx]))

                ki_scores.sort(key=lambda x: x[1])  
                clients[i].new_priors[j] = ki_scores[-1][0]

        # Ensure paralell update
        for i in range(num_clients):
            
            if clients[i].has_stopped:
                continue
            
            clients[i].priors = copy.copy(clients[i].new_priors)
            clients[i].n_sampled = copy.copy(clients[i].new_n_sampled)
            clients[i].local_model = copy.deepcopy(clients[i].new_model)

            not_i_idx = np.arange(num_clients) != i

            if(args.alt_tau):
                tau_new = tau_function(round, args.tau, c.DAC_VAR_WEIGHT)
            else:
                tau_new = args.tau

            clients[i].priors_norm[not_i_idx] = softmax_scale(
                np.clip(clients[i].priors[not_i_idx], a_min=c.DAC_MIN_SIMILARITY, a_max=c.DAC_MAX_SIMILARITY), tau_new)

        train_clients(round, clients, args.local_epochs)
        
    return clients