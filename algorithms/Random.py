import numpy as np

from algorithms.train_utils import *

def random(clients, args):
    neighbours = np.arange(args.num_clients)

    for round in range(args.n_rounds):
        for i in range(args.num_clients):
            
            if clients[i].has_stopped:
                continue
                
            sampled_idx = np.random.choice(neighbours, args.n_sampled, p=clients[i].priors, replace=False)
            model_weights = [clients[j].local_model.state_dict() for j in sampled_idx]
            nmb_datapoints = [len(clients[j].ldr_train.dataset) for j in sampled_idx]

            for j in sampled_idx: 
                clients[i].n_sampled[j] += 1

            model_weights.insert(0, clients[i].local_model.state_dict())
            nmb_datapoints.insert(0, len(clients[i].ldr_train.dataset))
        
            w_new = fed_avg(model_weights, nmb_datapoints)
            clients[i].new_model.load_state_dict(w_new)
        
        for i in range(args.num_clients):
            
            if clients[i].has_stopped:
                continue
            clients[i].local_model = copy.deepcopy(clients[i].new_model)   
        
        train_clients(round, clients, args.local_epochs, args.random_finetune)
            
    return clients