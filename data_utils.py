import numpy as np
import random

import algorithms.variables as v
import algorithms.constants as c

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

import copy


### Dataset Classes
class LocalDataset(Dataset):
    def __init__(self, dataset : Dataset, idxs : range, transform : transforms.Compose):
        self.dataset = dataset
        self.indices = list(idxs)
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.dataset[self.indices[index]]
        if self.transform != None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.indices)

    
# Counter class for used data
class ModuloInt:
    def __init__(self, value : int, modulo : int):
        self.value = value
        self.modulo = modulo
    
    def __str__(self):
        return str(self.value)
    
    def __add__(self, other):
        return ModuloInt((self.value + other) % self.modulo, self.modulo)


### Utility Functions
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def calculate_clients_per_cluster(num_clients : int, cluster_sizes : list[float]):
    
    clients_per_cluster = np.zeros(len(cluster_sizes), dtype=int)        

    # Store number of clients in each cluster
    for i, size in enumerate(cluster_sizes):
        count = int(size*num_clients)
        clients_per_cluster[i] += count

    # Add 1 to each cluster to allocate all clients to a cluster
    remainder = int(num_clients - np.sum(clients_per_cluster))
    for i in range(remainder):
        clients_per_cluster[i%len(cluster_sizes)] += 1
    
    v.clients_per_cluster = clients_per_cluster

def get_local_dataset(dataset_name : str, num_datapoints : int, transform : transforms, 
                      cluster_index : int = 0, time_step : int = -1, custom_name : str = None) -> LocalDataset:
    
    if custom_name == None:
        custom_name = dataset_name
    
    if v.experiment == 'covariate': cluster_index = 0
        
    start_index = v.label_ids_used[custom_name][cluster_index].value
    v.label_ids_used[custom_name][cluster_index] += num_datapoints
    end_index = v.label_ids_used[custom_name][cluster_index].value
    
    # Reset ModuloInt if override
    if end_index < start_index:
        modulo = v.label_ids_used[custom_name][cluster_index].modulo
        v.warning_message += (f'Timestep: {time_step}: \tReshuffling dataset ' 
                          + f'{dataset_name} for cluster index {cluster_index}!\n')
        
        ## First datapoints
        ind = list(range(start_index, modulo))
        ls = list(v.label_ids[custom_name][cluster_index][ind])
        
        np.random.shuffle(v.label_ids[custom_name][cluster_index])
        
        ## Second datapoints
        ind = list(range(0, num_datapoints - len(ls)))
        ls.extend(list(v.label_ids[custom_name][cluster_index][ind]))
    else:    
        ls  = list(v.label_ids[custom_name][cluster_index][start_index:end_index])
    
    return LocalDataset(v.datasets[dataset_name], ls, transform)    


### Initialization
def init_dataset(args) -> tuple[list[LocalDataset], list[LocalDataset], list[int], any]:
    
    if 'metadata' in args and 'dataset' in args.metadata:
        v.switch_datasets = True  
    
    calculate_clients_per_cluster(args.num_clients, args.cluster_sizes)
    
    load_datasets_to_memory(args)
    client_trainset, client_valset, client_cluster_ids = generate_datasets(args.num_clients, 
                                                                           args.n_data_train, 
                                                                           args.n_data_val)
    generate_testdata()
    
    return client_trainset, client_valset, client_cluster_ids, v.criterions


### Generate Datasets

def load_datasets_to_memory(args):
    
    if v.switch_datasets:
        datasets = args.metadata['dataset']
    else:
        datasets = [args.dataset]
        
    v.dataset_names = datasets
    
    for dataset in datasets:
        load_dataset_to_memory(dataset, train=True, download=True)
        load_dataset_to_memory(dataset, train=False, download=True)
        load_transforms(dataset)

    if args.conditional_probabilities:
        apply_label_swaps(args.n_clusters, args.dataset, args.conditional_numbs)

def apply_label_swaps(n_clusters, dataset, conditional_numbs):
    
    for i in range(n_clusters):
        ## Copy over default dataset
        set_name = f'{dataset}_conditional_{i}'
        v.dataset_names.append(set_name)
        v.datasets[set_name] = copy.deepcopy(v.datasets[dataset])
        
        test_name = f'{dataset}_conditional_{i}_test'
        v.datasets[test_name] = copy.deepcopy(v.datasets[f'{dataset}_test'])
        
        ## Skip if no label swap
        if len(conditional_numbs[i]) == 0:
            continue
        
        ## Get numbers which will be swapped
        int_1 = conditional_numbs[i][0]
        int_2 = conditional_numbs[i][1]
        
        # Switch labels for train/validation data
        indexes_1 = np.array(np.where(np.array(v.datasets[set_name].targets) == int_1)[0])
        indexes_2 = np.array(np.where(np.array(v.datasets[set_name].targets) == int_2)[0])
        
        for i in indexes_1:
            v.datasets[set_name].targets[i] = int_2
        for i in indexes_2:
            v.datasets[set_name].targets[i] = int_1

        # Switch labels for test data
        indexes_1 = np.where(np.array(v.datasets[test_name].targets) == int_1)[0]
        indexes_2 = np.where(np.array(v.datasets[test_name].targets) == int_2)[0]
        
        for i in indexes_1:
            v.datasets[test_name].targets[i] = int_2
        for i in indexes_2:
            v.datasets[test_name].targets[i] = int_1

def load_dataset_to_memory(dataset_name : str, train : bool = True, download : bool = False):
    
    train_dataset, criterion = load_dataset(dataset_name, train, download)
    
    if not train: dataset_name += '_test'
        
    v.label_ids[dataset_name] = []
    v.label_ids_used[dataset_name] = []
    
    # Find data index of where label group labels are
    if v.experiment == 'label':
        for group in v.label_groups:
            index = []
            for label_id in group:
                if 'svhn' in dataset_name:
                    idx = np.where(np.array(train_dataset.labels) == label_id)[0]
                else:
                    idx = np.where(np.array(train_dataset.targets) == label_id)[0]
                index.extend(idx)
            np.random.shuffle(index)
            v.label_ids[dataset_name].append(np.array(index))
            v.label_ids_used[dataset_name].append(ModuloInt(0, len(index)))
            
    elif v.experiment == 'covariate':
        dataset_len = len(train_dataset)
        index = np.arange(dataset_len)
        np.random.shuffle(index)

        v.label_ids[dataset_name].append(index)
        v.label_ids_used[dataset_name].append(ModuloInt(0, dataset_len))   

    v.datasets[dataset_name] = train_dataset
    v.criterions = criterion      

def load_dataset(dataset_name : str, train : bool, download : bool):
    if dataset_name == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(
            './datasets', train=train, download=download)
        criterion = torch.nn.NLLLoss()
    elif dataset_name == 'cifar100':
        train_dataset = torchvision.datasets.CIFAR100(
            './datasets', train=train, download=download)
        criterion = torch.nn.NLLLoss()
    elif dataset_name == 'fashion-mnist':
        train_dataset = torchvision.datasets.FashionMNIST(
            './datasets', train=train, download=download)
        criterion = torch.nn.NLLLoss()
    elif dataset_name == 'svhn':
        train_split = 'train' if train == True else 'test'
        train_dataset = torchvision.datasets.SVHN(
            './datasets', split=train_split, download=download)
        criterion = torch.nn.NLLLoss()
    elif dataset_name == 'mnist':
        train_dataset = torchvision.datasets.MNIST(
            './datasets', train=train, download=download)
        criterion = torch.nn.NLLLoss()
    elif dataset_name[0:4] == 'pacs':
        pacs_index = int(dataset_name[-1])
        image_folders = ['./datasets/PACS/photo', './datasets/PACS/art_painting', './datasets/PACS/cartoon', './datasets/PACS/sketch']
        file_name = image_folders[pacs_index]
        if not train: file_name += '_test'
        train_dataset = torchvision.datasets.ImageFolder(file_name)
        
        criterion = torch.nn.NLLLoss()
    
    return train_dataset, criterion

def generate_testdata():
    
    for dataset in v.dataset_names:
        v.testdata[dataset] = []
        
        for i, _ in enumerate(v.clients_per_cluster):
            
            if v.conditional_probabilities:
                name_index = i+1
                transform_index = 0
            else: 
                name_index = 0
                transform_index = 0
            
            dataset_name = v.dataset_names[name_index]
            transform_name = v.dataset_names[transform_index]
            
            if 'pacs' in dataset_name:
                test_points = c.TEST_PACS_NUM_DATAPOINTS
            else:
                test_points = c.TEST_NUM_DATAPOINTS
        
            transform = v.transforms[transform_name][i]  
            testdata = get_local_dataset(dataset_name = f'{dataset_name}_test', 
                                         num_datapoints = test_points, 
                                         transform = transform, 
                                         cluster_index = i, custom_name=f'{transform_name}_test')
            test_loader = DataLoader(testdata, 
                                batch_size=c.TEST_BATCHSIZE, shuffle=True, 
                                pin_memory=c.ENABLE_PIN_MEMORY, 
                                num_workers=c.WORKER_SIZE, worker_init_fn=seed_worker)
                
            v.testdata[dataset].append(test_loader)
            
def generate_datasets(num_clients : int, n_data_train : int, n_data_val : int) \
    -> tuple[list[LocalDataset], list[LocalDataset], list[int]]:
    
    client_trainset = np.empty(num_clients, dtype=LocalDataset)
    client_valset = np.empty(num_clients, dtype=LocalDataset)
    client_clusters = np.zeros(num_clients, dtype=int)
    client_index = 0
    
    for i, cluster_size in enumerate(v.clients_per_cluster):    
        
        # Sets correct dataset name
        if v.switch_datasets:
            name_index = i
            transform_index = i
        elif v.conditional_probabilities:
            name_index = i+1
            transform_index = 0
        else: 
            name_index = 0
            transform_index = 0
            
        dataset_name = v.dataset_names[name_index]
        transform_name = v.dataset_names[transform_index]
        
        transform = v.transforms[transform_name][i]     

        # For each cluster divide data into train/validation LocalDataset's
        for _ in range(cluster_size):
            
            client_trainset[client_index] = get_local_dataset(dataset_name, n_data_train, 
                                                              transform, cluster_index = i, custom_name = transform_name)
            client_valset[client_index] = get_local_dataset(dataset_name, n_data_val, 
                                                            transform, cluster_index = i, custom_name = transform_name)
            
            # Store each clients training/validation data and cluster identification
            client_clusters[client_index] = i   
            client_index += 1                           
            
    return client_trainset, client_valset, client_clusters

### Transforms
def load_transforms(dataset_name : str):
        
    if v.experiment == 'covariate':
        num_transforms = len(v.rotations_list)
    elif v.experiment == 'label':
        num_transforms = len(v.label_groups)
        
    v.transforms[dataset_name] = []
    for i in range(num_transforms):
        
        if v.experiment == 'covariate':
            index = i
        else:
            index = 0
            
        trans = get_transform(dataset_name=dataset_name, cluster_index=index)
        v.transforms[dataset_name].append(trans)

def get_transform(dataset_name : str, cluster_index : int = 0) -> transforms:
    
    trans = transforms.Compose([transforms.ToTensor()])
    insert_pos = 0
    
    # Set default number of channels based on loaded images
    if v.datasets[dataset_name][0][0].mode == 'RGB':
        num_channels = 3
    elif v.datasets[dataset_name][0][0].mode == 'L':
        num_channels = 1
    
    # Set default image size based on loaded images
    img_size = v.datasets[dataset_name][0][0].size[0]
    
    # Override num_channels if metadata provided
    if 'num_channels' in v.metadata:
        if v.metadata['num_channels'] == 1 and num_channels == 3:
            grayscale_transform = transforms.Grayscale(1)
            trans.transforms.insert(insert_pos, grayscale_transform)
            insert_pos += 1
            num_channels = 1
        elif v.metadata['num_channels'] == 3 and num_channels == 1:
            grayscale_transform = transforms.Grayscale(3)
            trans.transforms.insert(insert_pos, grayscale_transform)
            insert_pos += 1 
            num_channels = 1
    
    # Override image_size if metadata provided
    if 'img_size' in v.metadata:
        if v.metadata['img_size'] == 28 and img_size == 32:
            trans.transforms.insert(insert_pos, transforms.Resize((28, 28)))
        elif v.metadata['img_size'] == 32 and img_size == 28:
            trans.transforms.insert(insert_pos, transforms.Pad(2))
    
    # Normalization
    normal_transform = transforms.Normalize((0.5,)*num_channels, (0.5,)*num_channels)
    trans.transforms.append(normal_transform)
    
    # Rotations
    if v.experiment == 'covariate':
        rot_deg = v.rotations_list[cluster_index]
        if rot_deg != 0:
            trans_rotation = transforms.RandomRotation(degrees=(rot_deg, rot_deg))
            trans.transforms.append(trans_rotation)
    
    return trans


### New Data
def generate_new_data(num_datapoints : int, cluster_index : int = 0, time_step : int = -1) -> LocalDataset:
    
    if v.switch_datasets:
        dataset_name = v.metadata['dataset'][cluster_index]
    else:
        dataset_name = v.dataset_names[0]
    
    custom_name = dataset_name
    if v.conditional_probabilities:
        dataset_name = v.dataset_names[cluster_index+1]
    
    return get_local_dataset(dataset_name, 
                                num_datapoints, 
                                transform=v.transforms[custom_name][cluster_index], 
                                cluster_index=cluster_index, 
                                time_step=time_step, 
                                custom_name=custom_name)