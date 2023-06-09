import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import os
import sys
import shutil
import json

from run_experiments import*


# --------------------------------------------------------------------------------------------------------
#                                           SAVE/LOAD DATA
# --------------------------------------------------------------------------------------------------------

def load_clients(num_clients, path):
    clients = []
    if OLD_FILE:
        sys.modules['framework'] = sys.modules['algorithms']
    for idx in range(num_clients):
        pickle_path = path + f'client_{idx}.pkl'
        with open(pickle_path, 'rb') as filehandler:
            client = pickle.load(filehandler)
            clients.append(client)
    return clients

def save_df(df, save_path):
    df.to_csv(save_path + 'dataframe.csv', index=False)

def save_dict(dict, save_path):
    with open(save_path + '_config.txt', 'w') as filehandler:
        json.dump(dict, filehandler, indent=2)

def load_experiment_df(path, ref_config_dict = None, var = None):
    config_dict = check_config(path + '_config.txt', ref_config_dict, var = var, load=True)
    return pd.read_csv(path + 'dataframe.csv'), config_dict

def create_info_df(clients, ref_config_dict, columns, timeframe):
    data = pd.DataFrame()

    if timeframe == 'timestep':
        timesteps = np.arange(ref_config_dict['num_time_steps'])

        for client in clients:
            client_dict = {
                'timestep' : timesteps,
                'drift_true_detection' : client.drift_true_detections
            } 

            for column in columns:
                column_vals = getattr(client, column)
                if len(column_vals) != ref_config_dict['num_time_steps']:
                    raise Exception(f"Error: Encountered column {column} with wrong number of elements for TIMESTEP df")
                client_dict[column] =  getattr(client, column)
            client_df = pd.DataFrame(client_dict)
            data = pd.concat([data, client_df], axis = 0)

    elif timeframe == 'rounds':
        rounds = len(clients[0].val_acc_list)

        for client in clients:
            # Create list of cluster_ids over all rounds
            cluster_rounds = [client.cluster_id[0]] # To pad for initialization
            drift_true_detections = [client.drift_true_detections[0]]

            for timestep in range(ref_config_dict['num_time_steps']):
                cluster_rounds.extend([client.cluster_id[timestep]] * ref_config_dict['n_rounds'])
                drift_true_detections.extend([client.drift_true_detections[timestep]] * ref_config_dict['n_rounds'])
            cluster_rounds = map(str, cluster_rounds)

            client_dict = { # Always add rounds and cluster columns to timestep = 'rounds'
                'rounds' : np.arange(rounds),
                'cluster' : cluster_rounds,
                'drift_true_detection' : drift_true_detections
            }

            for column in columns:
                
                column_vals = getattr(client, column)
                if 'continual' in column:
                    for i, col in enumerate(column_vals):     
                        column_name =  f'{column}_cluster_{i}'
                        if len(col) != rounds:
                            raise Exception(f"Error: Encountered column {column_name} with wrong number of elements for ROUNDS df")
                        client_dict[column_name] = col
                else:
                    if len(column_vals) != rounds:
                        raise Exception(f"Error: Encountered column {column} with wrong number of elements for ROUNDS df")
                    client_dict[column] =  column_vals
            client_df = pd.DataFrame(client_dict)
            data = pd.concat([data, client_df], axis = 0)
    else:
        raise Exception(f"Error: You must select a valid timeframe when using extract_df, got {timeframe}")

    return data

# --------------------------------------------------------------------------------------------------------
#                                        UTILITIES
# --------------------------------------------------------------------------------------------------------

def check_config(config_path, ref_config_dict, var = None, load = False):
    with open(config_path) as f:
        run_config_txt = f.read()
    run_config_dict = json.loads(run_config_txt)
    
    if ref_config_dict is None:
        if not load:
            del run_config_dict['framework']
            del run_config_dict['seed']

        ref_config_dict = run_config_dict
    
    # else:
    #     for ref_key, ref_val in ref_config_dict.items():
    #         exp_val = run_config_dict[ref_key]
    #         if exp_val != ref_val:
    #             print(f"Warning: {ref_key} values does not match for {config_path}")
        
    return ref_config_dict

def make_path(path):
    exists = os.path.exists(path)

    if not exists:
        os.makedirs(path)
        print(f'New run directory {path} was created')

def get_path(experiment, framework, seed, var=None, var_val = None, folder = None):
    
    if var is not None:
        run_name = f'{experiment}_{framework}_{seed}_{var}_{var_val}'
    else:
        run_name = f'{experiment}_{framework}_{seed}'
    
    folder_dir = '/save/runs/'
    if folder is not None:
        folder_dir += f'{folder}/'
    
    return os.getcwd() + folder_dir + run_name + '/'

def get_analysis_save_path(experiment, framework, seeds, var = None):
    if var is not None:
        exp_name = f'{experiment}_{framework}_{str(seeds)[1:-1]}_{var}'
    else:
        exp_name = f'{experiment}_{framework}_{str(seeds)[1:-1]}'

    save_path = os.getcwd() + '/save/analysis/' + exp_name + '/'
    make_path(save_path)
    return save_path, exp_name

def get_df(path, ref_config_dict, columns, timeframe, var):
    ref_config_dict = check_config(path + '_config.txt', ref_config_dict, var)
    print(f'Loading clients from run {path}')
    clients = load_clients(ref_config_dict['num_clients'], path)
    data = create_info_df(clients, ref_config_dict, columns, timeframe)
    return data, ref_config_dict

def get_framework_df(columns, ref_config_dict, timeframe, experiment, framework, seeds, var = None, var_vals = None, folder = None):
    framework_dataframe = pd.DataFrame()
    move_folder_dir = []
    for seed in seeds:
        if var is not None:
            run_dataframe = pd.DataFrame()
            for var_val in var_vals: 
                run_path = get_path(experiment, framework, seed, var, var_val, folder)
                move_folder_dir.append(run_path)
                tmp_dataframe, ref_config_dict = get_df(run_path, ref_config_dict, columns, timeframe, var = var)

                tmp_dataframe[var] = str(var_val)
                run_dataframe = pd.concat([run_dataframe, tmp_dataframe])

        else:
            run_path = get_path(experiment, framework, seed, folder=folder)
            run_dataframe, ref_config_dict = get_df(run_path, ref_config_dict, columns, timeframe, var = var)
            
        run_dataframe['framework'] = framework
        run_dataframe['seed'] = str(seed)
        framework_dataframe = pd.concat([framework_dataframe, run_dataframe])

    return framework_dataframe, ref_config_dict

def move_folder(source):
    
    indexes = [i for i, letter in enumerate(source) if letter == '/']
    target = source[0:indexes[-3]] + '/_analyzed' + source[indexes[-2]:]
    make_path(target)
    
    file_names = os.listdir(source)
    
    for file_name in file_names:
        shutil.move(os.path.join(source, file_name), target)
    
    os.rmdir(source)
    

# --------------------------------------------------------------------------------------------------------
#                                      ANALYSIS FUNCTIONS
# --------------------------------------------------------------------------------------------------------

def compare_runs(runs, run_names, timeframe = 'rounds', data_columns = ['val_acc_list', 'val_loss_list', 'train_loss_list'], column_names = ['val acc', 'val loss', 'train loss'], folder=None, comparison_name = '__Comparison___'):
    if vars is not None:
        if folder is not None:
            ref_config_dict = check_config(os.getcwd()+f'/save/runs/{folder}/{runs[0][0]}/_config.txt', None, var = None, load = False)
        else:
            ref_config_dict = check_config(os.getcwd()+f'/save/runs/{runs[0][0]}/_config.txt', None, var = None, load = False)
        
    else:
        ref_config_dict = None
    
    tot_df = pd.DataFrame()

    for idx in range(len(runs)):
        seed_df = pd.DataFrame()
        for seed_run in runs[idx]:
            if folder is not None:
                run_path = os.getcwd() + f'/save/runs/{folder}/{seed_run}/'
            else:
                run_path = os.getcwd() + f'/save/runs/{seed_run}/'
            data, ref_config_dict = get_df(run_path, ref_config_dict, columns, timeframe, var = None)
            seed_df = pd.concat([data, seed_df])

        seed_df = seed_df.groupby(['rounds']).mean(numeric_only=True)
        seed_df['framework'] = run_names[idx]
        tot_df = pd.concat([seed_df, tot_df])
    
    tot_df.rename(columns= dict(zip(data_columns, column_names)), inplace=True)


    for column in column_names:
        plot_comparison([tot_df], column, ref_config_dict, timeframe, title=comparison_name)

def print_experiment_scores(runs, run_names, folder=None):
    timeframe = 'rounds'
    columns = ['best_test_acc_list', 'val_acc_list']
    column_names = ['Test_acc', 'Val_acc']
    args_dict = None
    
    tot_df = pd.DataFrame()

    for idx in range(len(runs)):
        seed_df = pd.DataFrame()
        for seed_run in runs[idx]:
            if folder is not None:
                run_path = os.getcwd() + f'/save/runs/{folder}/{seed_run}/'
            else:
                run_path = os.getcwd() + f'/save/runs/{seed_run}/'
            data, args_dict = get_df(run_path, args_dict, columns, timeframe, var = None)
            seed_df = pd.concat([data, seed_df])

        seed_df = seed_df.groupby(['rounds']).mean(numeric_only=True)
        seed_df['framework'] = run_names[idx]
        tot_df = pd.concat([seed_df, tot_df])
    
    tot_df.rename(columns= dict(zip(columns, column_names)), inplace=True)

    num_timesteps = args_dict['num_time_steps']
    num_rounds = args_dict['n_rounds']

    end_of_timestep_accs = [0] * num_timesteps
    num_rounds_until_threshold = [0] * num_timesteps
    drop = [0] * num_timesteps
    test_column = 'Val_acc'

    for framework in run_names:
        tmp_df = tot_df.loc[tot_df['framework'] == framework]
        print(f'Values for framework: {framework}')

        for timestep in range(1, num_timesteps+1):
            timestep_round = (num_rounds * timestep)
            end_of_timestep_accs[timestep-1] = (tmp_df[test_column].iloc[[timestep_round]].item())
        

        for timestep in range(1, num_timesteps):
            start_of_timestep = timestep * num_rounds
            start_of_timestep_val = tmp_df[test_column].iloc[[1 + start_of_timestep]].item()
            end_of_prev_timestep_val = end_of_timestep_accs[timestep]
            diff = end_of_prev_timestep_val - start_of_timestep_val
            threshold = start_of_timestep_val + 0.9*(diff)
            lowest_value = 100

            for round_idx in range(1, num_rounds+1):
                curr_val = tmp_df[test_column].iloc[[round_idx + start_of_timestep]].item()
                if  curr_val < lowest_value:
                    lowest_value = curr_val
            drop[timestep] = end_of_prev_timestep_val - lowest_value

            for round_idx in range(1, num_rounds+1):
                if tmp_df[test_column].iloc[[round_idx + start_of_timestep]].item() >= threshold:
                    num_rounds_until_threshold[timestep] = round_idx
                    break
        
        print(f'Acc: {sum(end_of_timestep_accs)/num_timesteps}')
        print(f'Threshold rounds: {sum(num_rounds_until_threshold)/(num_timesteps-1)}')
        print(f'Drop: {sum(drop)/(num_timesteps-1)}')


# --------------------------------------------------------------------------------------------------------
#                                             PLOTS
# --------------------------------------------------------------------------------------------------------

def plot_comparison(dfs, y_label, args_dict, timeframe, var = None, title = "Comparison"):
    
    save_path = os.getcwd() + '/save/analysis/' + title + '/'

    sns.set()
    plt.figure(figsize = (12, 10))
    sns.set(font_scale=2)

    palette = {
        "DAC":"blue",
        "DAC w finetune" : "slategrey",
        "Random w finetune" : "orchid",
        "Random":"tab:orange", 
        "HAST (Depth 1)":"slategrey",
        "HAST (Depth 2)":"orchid",
        "HAST (Depth 3)":"green",
        "HAST": "green",
        "HAST (Depth 4)":"dodgerblue",
        }
    
    dashes = {
        "DAC":"",
        "Random":"", 
        "DAC w finetune" : "",
        "Random w finetune" : "",
        "HAST (Depth 1)":(2,2),
        "HAST (Depth 2)":(2,2),
        "HAST (Depth 3)":"",
        "HAST":"",
        "HAST (Depth 4)":(2,2),
        }

    if timeframe == 'rounds':
        for timestep in range(1, args_dict['num_time_steps']):
            timestep_round = 1 + (args_dict['n_rounds'] * timestep)
            plt.axvline(timestep_round, linestyle = "--", color = 'gray')

    total_df = pd.DataFrame()
    for df in dfs:
        if var is not None and var not in df.columns:
            df[var] = df['framework'].iloc[0]
        total_df = pd.concat([total_df, df])

    if var is not None:
        sns.lineplot(data = total_df, x = timeframe, y = y_label, hue= var)
    else:
        #sns.lineplot(data = total_df, x = timeframe, y = y_label, hue= 'framework', palette = palette, style='framework', dashes=dashes)
        sns.lineplot(data = total_df, x = timeframe, y = y_label, hue= 'framework')
    
    # plt.title(f"{title} : {y_label}")
    plt.xlabel('Rounds')
    plt.legend(title='Method')

    make_path(save_path)
    plt.savefig(save_path + y_label.replace(" ", "_") + '.png')
    save_df(df, save_path)
    save_dict(args_dict, save_path)


# --------------------------------------------------------------------------------------------------------
#                                             MAIN
# --------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    runs = [
        ['test_HA_DAC_3'],
        ['test_random_3'],
    ]

    run_names = [
        'HAST',
        'Random'
    ]

    name = "test_analyze"
    OLD_FILE = True

    
    columns = ['train_loss_list',
               'val_loss_list',
               'val_acc_list', 
               'best_test_loss_list', 
               'best_test_acc_list',
               'early_stopped']
    
    column_names = [
        'Train loss',
        'Val loss',
        'Val acc',
        'Test loss',
        'Test acc',
        'Early stopped'
    ]

    folder = '_unanalyzed'

    print_experiment_scores(runs, run_names, folder = folder)
    compare_runs(runs, run_names, data_columns=columns, column_names = column_names, folder=folder, comparison_name=name)
