Repository for master thesis [Decentralized Deep Learning under
Distributed Concept Drift](), Chalmers Spring 23.

Contributors (Equal): 
- Emilie Klefbom [EmilieKar](https://github.com/EmilieKar)
- Marcus Örtenberg Toftås [MackanT](https://github.com/MackanT)
---
# HAST Hierarchical Aggregation with Similarity based Tuning

$\mathtt{HAST}$, is a decentralized deep learning algorithms that can adapt to spacial and temporal data variations in a peer-to-peer network. It combines does this by using a hierarchical sampling and aggregation scheme. At its core, it is based on the principle of transfer learning where we belive that some domain knowledge is shared between tasks, therefore training a common feature extractor is possible. To better capture the nuances of the separate tasks we can then fine-tune a classifier based on similar clients, using similarity clustering taken from the [DAC](https://github.com/edvinli/DAC) algorithm.

In more formal terms $\mathtt{HAST}$ works as follows. Each client $k$ has a neural network consisting of a feature extractor $f^k_\theta$ and a classifier $f^k_\phi$ (the whole model being $f^k_\Omega$ = $f^k_\phi \circ f^k_\theta$). $\mathtt{HAST}$ then consists of three steps. For each client $k$ and for each communication round: 


- A subset of clients $S_{rand}$ is sampled uniformly and at random. Client $k$ updates all layers $f^k_\Omega$ using $\mathtt{FedAvg}$.
- A subset of clients $S_{sim}$ is randomly sampled based on their similarity to client $k$. Client $k$ updates \textit{only} the classifier layers $f^k_\phi$ using $\mathtt{FedAvg}$.
- Client $k$ performs local training on its own data, updating the whole model $f^k_\Omega$. Afterward, it fine-tunes only the classifier $f^k_\phi$.


The similarity function is the empirical training loss of client $C_k$'s model $w_k$ on client $C_j$'s dataset $D_j$: $s_{kj} = 1/\mathcal{L}(w_k;D_j)$. This similarity score is transformed using a softmax with a temperature scaling $\tau$ in order to get a probability vector $p_{kj}$ for each client and each communication round. A two-hop scheme is also used to allow clients to estimate similarities with clients that have not yet communicated which increases convergence speed.
# Repository information
## Required packages 

    torch
    numpy
    pandas
    seaborn
    pickle
    git

## Algorithms
---
This folder code for algorithm used in our report is included as well as an algorithm called DAC2 which is an updated version of DAC using a similarity based aggregation method. It outperforms the original DAC algorithm but as HAST performance was even better it wasn't included in the report.

## Run_experiments.py
----
Experiments can be run by changing the experiments set in main. Experiments are predefined sets of parameters defined in utilities.py 

For example the present eperiment looks as follows


    experiments = ['cifar10_2_cluster_label_drift_random']
    frameworks = ['HAST']
    seeds = [0]

this will run HAST on the ezperiment cifar10_2_cluster_label_drift_random which will load in the following parameters from utils. More in depth description of all parameters and available experiments can be found in utils. 

    "n_rounds" : 200,
    "num_time_steps": 10,
    "bs" : 8,
    "local_epochs" : 3,
    "n_data_train" : 250,
    "n_data_val": 75,
    "dataset": "cifar10",
    "num_classes": 10,
    "iid": "label",
    "label_groups_cifar10" : [[0,1,8,9],[3,4,5,7]],
    "label_groups_mnist" : [[0,1,8,9],[3,4,5,7]],
    "label_groups_pacs" : [[0,1,2,3,4,5,6],[0,1,2,3,4,5,6]],
    "random_lr": 0.00008,
    "DAC_lr": 0.00008,
    "HAST_lr": 0.00008, 
    "framework": 'HAST',
    "n_sampled": 3,
    "tau": 30,
    "hierarchichal_aggregation_depth": 3,
    "hierarchichal_random_aggregation_all_layers": True,
    "HAST_finetune": True,

To run more than one algorithm just add the names to the framework list. 

    framework = ['HAST', 'random', 'DAC', 'DAC2']

To run more than one seed just add the desired seeds to the seed list 

    seed = [42,11,27]

When experiment is done the clients will be saved to a folder by the experiment_seed_algorithm name and these client folders are then called on in analyze_results. 

When all parameters are set you simply run the python file to run the selected experiment(s)

    python run_experiments.py

## Analyze_results
---
The parameters in analyze results should be set in the following fashion. 


    runs = [
        [ 'folder1_alg1',
          'folder1_alg1',
          'folder1_alg1',
          ...],
        [ 'folder1_alg2',
          'folder1_alg2',
          ...],
        ...
    ]

    run_names = [
        'alg1',
        'alg2',
        ...
    ]

    name = "folder_name_for_plots"

All files in the same inner list in runs will be averaged and plotted as one line by the name of the corresponding index in run_names. By default the one plot will be created for each column in columns and the plot axis will be names according to the column name parameter. Columns read out lists stored on clients. 

When all parameters are set you simply run the python file to analyze the selected experiment(s)

    python analyze_results.py