# -*- coding: utf-8 -*-
"""
@author: ---
"""

from main_functions_graft_lab2 import *
import itertools
import numpy as np
import os

#%% FUNCTIONS

full_dict = r'cond_data_bump_new.npy' 
# this data was created with the aid of the jupyter notebook from the neuralatent benchmark
# https://github.com/neurallatents/neurallatents.github.io/blob/master/notebooks/area2_bump.ipynb
data_bump = np.load(full_dict, allow_pickle=True).item()
window_num = 60
angle = 0
bin_size = 4

"""
specific angle - passive vs active
"""
def sum_bin(df, wind = 3, w_gaussian = False):
    """
    if overlap == 0:
    overlap = bin_size - 1
    Calculate the summed values for each column of a 2D array or DataFrame.
    
    Args:
        df (numpy.ndarray or pandas.DataFrame): The input data as a 2D array or DataFrame.
        wind (int): The window size used for summation. Defaults to 3.
    
    Returns:
        numpy.ndarray: A 2D array containing the summed values for each column.
    
    """

    if not isinstance(df, np.ndarray):
        df = df.values
    if w_gaussian:
        df_summed = np.vstack([np.sum(df[np.max([0,i-wind]):np.min([df.shape[0], i+wind]),:], 0).reshape((1,-1)) 
                               for  i in range(wind, df.shape[1] - wind)])
    else:
        df_summed = np.vstack([np.sum(df[np.max([0,i-wind]):np.min([df.shape[0], i+wind]),:], 0).reshape((1,-1)) * 2*np.exp(-0.3*vals_centered(np.max([0,i-wind]), np.min([df.shape[0], i+wind])))
                               for  i in range(wind, df.shape[1] - wind)])        
    return df_summed
    
def vals_centered(a,b):
    """
    Calculate the centered values between a and b.

    Parameters:
        a (int): Starting value.
        b (int): Ending value.

    Returns:
        ndarray: Array of centered values.

    """    
    x = np.arange(a,b)
    x = x - np.mean(x)
    return x

                           
        
def labels_to_nums(labels):
    """
    Convert a list of labels into numerical values.
    
    Args:
        labels (list or array-like): The input list of labels.
    
    Returns:
        tuple: A tuple containing a dictionary that maps numerical values to labels and a list of numerical values corresponding to the input labels.
    
    """
    dict_nums_labels ={}
    list_nums = []
    unique_labels = np.unique(labels)
    match_nets = np.arange(len(unique_labels)) + 1
    dict_match = dict(zip(match_nets, unique_labels ))
    dict_match_inverse = dict(zip(unique_labels , match_nets))      
    list_nums = [dict_match_inverse[label]  for label in labels]
    return dict_match, list_nums


def lists2list(xss)    :
    """
    Flatten a list of lists into a single list.

    Args:
        xss (list): The input list of lists.

    Returns:
        list: A flattened list containing all elements from the input lists.

    """    
    return [x for xs in xss for x in xs]         
        

take_only_active = True
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%% take both active and passive
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
if not take_only_active:
    labels = data_bump.keys()
    
    store_labels = []                                   # for storing labels 
    store_vals = []                                     # for storing values
    labels = []                                         # storing labels include both angle and perturbation                                                   
    for key,df_spec in data_bump.items():               # each key is (boolean, angle) tuple
        for angle in np.arange(0,316,45):               # go over all angles                    
            if key[1] == angle:                         # # identify the angles
                unique_trials = np.unique(df_spec['trial_id'])   # take the relevant df
                for trial in unique_trials:                      # find unique trials
    
                    cur_df = df_spec.iloc[df_spec['trial_id'].values == trial,:]['spikes'] # take only the relevant trial
                    cur_df = sum_bin(cur_df, wind = bin_size)[:window_num,:].T # count spikes
                    store_vals.append(cur_df)
                    store_labels.append(key[0])
                    labels.append((key))
                
    """
    SAVE RESULTS
    """
    store_vals = np.dstack(store_vals)           

    data_name = 'neuro_bump'            
    store_vals = np.dstack(store_vals)            
    dict_nums, list_nums = labels_to_nums(labels)
    labels_name = create_data_name(data_name = data_name,  type_name = 'labels')
    data_name_save = create_data_name(data_name = data_name,  type_name = 'data')
    np.save(labels_name, np.array(list_nums))  
    np.save(data_name_save,store_vals)

    store_vals = np.transpose(store_vals, [2,0,1])
    np.save('neuro_results_angles.npy', {'data':store_vals, 'labels':np.array(store_labels)})
    
    fig, ax = plt.subplots(2, 10, sharex = True, sharey = True)
    
    count_labs = {False: 0, True:0}
    
    for i in range(store_vals.shape[2]):
        if store_labels[i] == False and count_labs[False] < 10:
    
            sns.heatmap(store_vals[:,:,i], ax = ax[0,count_labs[False]], vmin = -5, vmax  = 5);count_labs[False] += 1
        elif store_labels[i] == True and count_labs[True] < 10:
            print('tttttt')
            sns.heatmap(store_vals[:,:,i], ax = ax[1,count_labs[True]], vmin = -5, vmax  = 5);count_labs[True] += 1
    [ax_s.set_xlabel('time') for ax_s in ax.flatten()]; [ax_s.set_ylabel('neuron') for ax_s in ax.flatten()]
    ax[0,0].set_ylabel('Active', fontsize = 20)
    ax[1,0].set_ylabel('Passive', fontsize = 20)
    fig.tight_layout()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#%% ONLY TAKE ACTIVE TRIALS
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
labels = data_bump.keys()

store_labels = []                                   # for storing labels 
store_vals = []                                     # for storing values
labels = []                                         # storing labels include both angle and perturbation
for key,df_spec in data_bump.items():               # each key is (boolean, angle) tuple
    for angle in np.arange(0,316,45):               # go over all angles
        if key[1] == angle and key[0] == False:     # identify the angles and take only if active (no perturb)
            unique_trials = np.unique(df_spec['trial_id'])  # take the relevant df
            for trial in unique_trials:                     # find unique trials

                cur_df = df_spec.iloc[df_spec['trial_id'].values == trial,:]['spikes']  # take only the relevant trial
                cur_df = sum_bin(cur_df, wind = bin_size)[:window_num,:].T              # count spikes
                store_vals.append(cur_df)
                store_labels.append(key[0])
                labels.append((key))
                
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%                
data_name = 'neuro_bump_angle'            


store_vals = np.dstack(store_vals)            
dict_nums, list_nums = labels_to_nums(labels)
labels_name = create_data_name(data_name = data_name,  type_name = 'labels')
data_name_save = create_data_name(data_name = data_name,  type_name = 'data')
np.save(labels_name, np.array(list_nums))  
np.save(data_name_save, store_vals)


np.save('neuro_results_angles_only_active%s.npy'%data_name, {'data':store_vals, 'labels':np.array(store_labels)})


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%   
# ANLY "False" - i.e. active, different angles.
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
labels = data_bump.keys()

store_labels = []
store_vals = []
labels = []
angles = np.arange(0,316,45)
angles_dict = {angle:0 for angle in angles}
max_angle_count = 10
for key,df_spec in data_bump.items():
    for angle in np.arange(0,316,45):
        if key[1] == angle and key[0] == False and angles_dict[angle] < max_angle_count:
            unique_trials = np.unique(df_spec['trial_id'])
            for trial in unique_trials:
     
                cur_df = df_spec.iloc[df_spec['trial_id'].values == trial,:]['spikes']
 
                cur_df = sum_bin(cur_df, wind = bin_size)[:window_num,:].T
                store_vals.append(cur_df)
                store_labels.append(key[0])
                labels.append((key))
                angles_dict[angle] += 1


"""
uneq_trials
"""

labels_angle = np.vstack(labels)[:,1]
dict_nums, list_nums = labels_to_nums(labels_angle)
list_nums = np.array(list_nums)
store_vals = np.dstack(store_vals)  
data_name = 'neuro_bump_angle_active_uneq_trials'    
np.save('grannet_%s_results_march_2023.npy'%data_name, {'label_dict_nums':dict_nums, 'labels':labels,
                                                        'Y':store_vals, 'labels_angle': labels_angle,
                                                        'dict_nums':dict_nums, 'labels_nums':list_nums})

labels_name = create_data_name(data_name = data_name,  type_name = 'labels')
data_name_save = create_data_name(data_name = data_name,  type_name = 'data')
np.save(labels_name, list_nums)  
np.save(data_name_save, store_vals)

"""
eq_trials
"""
min_trials = np.min(np.unique(list_nums, return_counts=True)[1])

list_nums_bool_include = []
dict_nums_store_counts = {el:0 for el in np.unique(list_nums)}
for label_num in list_nums:
    if dict_nums_store_counts[label_num] < min_trials:
        list_nums_bool_include.append(True)
    else:
        list_nums_bool_include.append(False)
    dict_nums_store_counts[label_num] = dict_nums_store_counts[label_num] + 1
    
list_nums = list_nums[np.array(list_nums_bool_include)]    
labels_angle = labels_angle[np.array(list_nums_bool_include)]     

labels = np.array(labels)[np.array(list_nums_bool_include)]     
store_vals  = store_vals[:,:,np.array(list_nums_bool_include)]  

data_name = 'neuro_bump_angle_active'    
np.save('grannet_%s_results_march_2023.npy'%data_name, {'label_dict_nums':dict_nums, 'labels':labels,
                                                        'Y':store_vals, 'labels_angle': labels_angle,
                                                        'dict_nums':dict_nums, 'labels_nums':list_nums})

labels_name = create_data_name(data_name = data_name,  type_name = 'labels')
data_name_save = create_data_name(data_name = data_name,  type_name = 'data')
np.save(labels_name, list_nums)  
np.save(data_name_save, store_vals)


    
    