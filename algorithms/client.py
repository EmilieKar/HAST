import numpy as np
import copy

import torch
from torch.utils.data import DataLoader

from algorithms.local_models import *
from data_utils import seed_worker
import algorithms.constants as c
import algorithms.variables as v
             
class Client(object):
    def __init__(self, train_data, val_data, **kwargs):
        self.device = kwargs['device']
        self.criterion = kwargs['criterion']
        self.lr = kwargs['lr']
        n_clients = kwargs['num_clients']
        self.n_clusters = kwargs['num_clusters']
        self.dataset = kwargs['dataset']
        self.batch_size = kwargs['batch_size']
        self.cluster_id = [kwargs['cluster_id']]
        self.total_rounds = kwargs['total_rounds']
        self.current_round = -1
        
        self.ldr_val = []
        self.create_data_loaders(train_data, val_data)

        self.received_models = []
        self.train_loss_list = np.zeros(self.total_rounds+1, dtype=float)
        self.val_loss_list = np.zeros(self.total_rounds+1, dtype=float)
        self.val_acc_list = np.zeros(self.total_rounds+1, dtype=float)
        self.best_val_loss_list = np.zeros(self.total_rounds+1, dtype=float)
        self.best_val_acc_list = np.zeros(self.total_rounds+1, dtype=float)
        
        self.best_test_loss_list = np.zeros(self.total_rounds+1, dtype=float)
        self.best_test_acc_list = np.zeros(self.total_rounds+1, dtype=float)
        
        self.early_stopped = np.zeros(self.total_rounds+1, dtype=float)
        self.has_stopped = False
        self.patience_counter = 0

        self.n_sampled = np.zeros(n_clients)
        self.best_val_loss = np.inf
        self.best_val_acc = -np.inf

        self.priors = np.zeros(n_clients)
        self.priors_norm = np.zeros(n_clients)

        self.new_priors = np.zeros(n_clients)
        self.new_n_sampled = np.zeros(n_clients)

    def initialize_model(self, model):
        self.local_model = copy.deepcopy(model)
        self.best_model = copy.deepcopy(model)
        self.new_model = copy.deepcopy(model)
        
        # Generate finetuned model with specific training params
        self.finetuned_model = copy.deepcopy(model)
        all_keys = model.state_dict().keys()
        if v.hierarchichal_aggregation_depth == 0:
            keys = all_keys
        else:
            keys = list(all_keys)[0:-v.hierarchichal_aggregation_depth]
        for name, param in self.finetuned_model.named_parameters():
            if name in keys:
                param.requires_grad = False
        
    def reinitialize_model(self, model):
        self.local_model = copy.deepcopy(model)

    def create_data_loaders(self, train_data, val_data):
        
        self.ldr_train = (DataLoader(train_data, 
                                    batch_size=self.batch_size, shuffle=True, 
                                    pin_memory=c.ENABLE_PIN_MEMORY, 
                                    num_workers=c.WORKER_SIZE, worker_init_fn=seed_worker))
        
        self.ldr_val.append(DataLoader(val_data, 
                                  batch_size = c.TEST_BATCHSIZE, shuffle=False, 
                                  pin_memory=c.ENABLE_PIN_MEMORY, 
                                  num_workers=c.WORKER_SIZE, worker_init_fn=seed_worker))


    def train(self,num_epochs, finetune=False):
        
        self.current_round += 1
        
        ## Early stopping -> skip training
        if self.has_stopped:
            
            self.train_loss_list[self.current_round] = self.train_loss_list[self.current_round - 1]
            
            self.val_loss_list[self.current_round] = self.val_loss_list[self.current_round - 1]
            self.val_acc_list[self.current_round] = self.val_acc_list[self.current_round - 1]
            
            self.best_test_loss_list[self.current_round] = self.best_test_loss_list[self.current_round - 1]
            self.best_test_acc_list[self.current_round] = self.best_test_acc_list[self.current_round - 1]
            
            for i in range(self.n_clusters):
                self.best_continual_loss_list[i, self.current_round] = self.best_continual_loss_list[i, self.current_round - 1]
                self.best_continual_acc_list[i, self.current_round] = self.best_continual_acc_list[i, self.current_round - 1]
            
            self.best_val_loss_list[self.current_round] = self.best_val_loss_list[self.current_round - 1]
            self.best_val_acc_list[self.current_round] = self.best_val_acc_list[self.current_round - 1]
            
            self.early_stopped[self.current_round] = 1

            return self.best_model, self.train_loss_list[self.current_round], self.best_val_loss_list[self.current_round - 1], self.best_val_acc_list[self.current_round - 1]
        
        self.local_model.train()
                
        optimizer = torch.optim.Adam(self.local_model.parameters(),lr=self.lr) #  weight_decay=0.001 Option to use something along these lines
        
        # Initial n-1 Epochs
        for _ in range(num_epochs-1):
            for data, labels in self.ldr_train:
                data, labels = data.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                log_probs = self.local_model(data.float())
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()

        # Final Epoch
        last_epoch_train_loss = 0.0
        num_batches = 0
        for data, labels in self.ldr_train:
            data, labels = data.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            log_probs = self.local_model(data.float())
            loss = self.criterion(log_probs, labels)
            loss.backward()
            optimizer.step()
            last_epoch_train_loss += loss.item()
            num_batches += 1

        last_epoch_train_loss = last_epoch_train_loss/num_batches
        self.train_loss_list[self.current_round] = last_epoch_train_loss
        
        
        # (Finetune and) Update best model
        if finetune:
            val_loss, val_acc = self.finetune(num_epochs)
        else:
            val_loss, val_acc = self.validate(self.local_model, train_set = False)
            
        self.val_loss_list[self.current_round] = val_loss
        self.val_acc_list[self.current_round] = val_acc
        
        if(val_loss < self.best_val_loss):
            self.best_val_loss = val_loss
            self.best_val_acc = val_acc
            self.best_model.load_state_dict(copy.deepcopy(self.local_model.state_dict()))
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        self.best_val_loss_list[self.current_round] = self.best_val_loss
        self.best_val_acc_list[self.current_round] = self.best_val_acc
            
        return self.best_model, last_epoch_train_loss, self.best_val_loss, self.best_val_acc
    

    def finetune(self, num_epochs):
        
        self.finetuned_model.load_state_dict(self.local_model.state_dict())
        current_loss, current_acc = self.validate(self.local_model)
        
        ## If hierarchichal_aggregation_depth = 0 -> only random communication -> no similarity finetuning
        if v.hierarchichal_aggregation_depth == 0:
            return current_loss, current_acc
        
        self.finetuned_model.train()
        optimizer = torch.optim.Adam(self.finetuned_model.parameters(),lr=self.lr)
        
        for _ in range(num_epochs):
            
            for data, labels in self.ldr_train:
                data, labels = data.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                log_probs = self.finetuned_model(data.float())
                loss = self.criterion(log_probs, labels)
                loss.backward()
                optimizer.step()
                
        loss, acc = self.validate(self.finetuned_model)
            
        if loss < current_loss:
            self.local_model.load_state_dict(self.finetuned_model.state_dict())
            current_loss = loss
            current_acc = acc
        
        return current_loss, current_acc
    
    
    def validate(self, model = None, train_set = False, dataloader = None, best_model = False):
        
        if dataloader is None:
            if train_set:
                dataloader = self.ldr_train
            else:
                dataloader = self.ldr_val[-1]
        
        correct = 0
        num_datapoints = 0
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():

            if model is None:
                if best_model:
                    model = self.best_model
                else:
                    model = self.local_model
            
            model.to(self.device)
            model.eval()
            
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                log_probs = model(inputs)
                _, predicted = torch.max(log_probs.data, 1)
                                            
                loss = self.criterion(log_probs,labels.long())                
                total_loss += loss.item()
                num_batches += 1

                num_datapoints += labels.size(0)
                correct += (predicted == labels).sum().item()

            val_acc = 100 * correct / num_datapoints
            val_loss = total_loss/num_batches

        return val_loss, val_acc
 