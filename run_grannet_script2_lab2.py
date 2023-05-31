# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 15:12:10 2022

@author: ---
"""
import os
"""
from Create_Synthetic_poisson import *
"""



from main_functions_graft_lab2 import *
from datetime import datetime as datetime2
ss = int(str(datetime2.now()).split('.')[-1])
seed = ss # np.random.randint(ss) # 0,aaa[50])l
np.random.seed(seed)



type_grannet = 'trends_grannet' 
to_recreate = False

params_full = params_default




if type_grannet.startswith('synth_'): 
    params_full['p'] = 10
    full_A, full_phi, additional_return = run_GraFT(data = 'data_synth_grannet_xmin_0_xmax_n_ymin_0_ymax_n.npy', corr_kern = []  ,
          params = params_full , grannet=True, images = False, data_type = 'synth_grannet') 
 
elif type_grannet == 'neuro_bump_angle_active' or type_grannet == 'neuro_bump_angle_active_minmax':
    params_full['inverse_params'] = {'T_inverse':2}
    n_zer =  2 #np.random.randint(2,3)
    params_full['n_neighbors'] = n_zer + 5
    params_full['hard_thres_params'] = {'non_zeros': n_zer,
                                       'thres_error_hard_thres': 10,  'keep_last':False,
                                       'T_hard': 1}
    params_full['grannet_params']['rounded'] = True 
    params_full['grannet_params']['rounded_max'] = 360    
    params_full['is_trials'] = True
    params_full['phi_only_dec'] = True
    params_full['noise_stuck_params'] = {'max_change_ratio': 0.01, 'in_a_row': np.random.randint(2,4), 
                                         'std_noise_A':0.25, 'std_noise_phi': 0.5, 'change_step': 20 }
    params_full['A_only_dec'] = True
    type_kernel = 'shared'
    params_full['is_trials_type'] = type_kernel
    params_full['epsilon'] = np.abs(np.random.randn()*0.06 + 2.1)
    params_full['beta'] = np.abs(np.random.rand()*0.01 + 0.02)
    params_full['zeta'] =  np.random.rand()+ 10
    params_full['weight_sim_nets'] =  np.random.randint(18,30)

    params_full['condition_unsupervised'] = False
    params_full['l4'] = 0.3

    n_nets =5

    if n_nets =='loop':
        rrr = str(np.random.randint(1,10000000))
        
        for j in range(3,15,3):
            params_full['name_addition']  = rrr + '_' + str(j)
            params_full['p'] = j
            params_full['max_learn'] = 300
            params_full['is_trials_type'] = type_kernel
            full_A, full_phi, additional_return = run_GraFT(data = 'data_%s_xmin_0_xmax_n_ymin_0_ymax_n.npy'%type_grannet, corr_kern = [],  
                  params = params_full , grannet=True, images = False, data_type = type_grannet , 
                  labels_name = 'labels_%s_xmin_0_xmax_n_ymin_0_ymax_n.npy'%type_grannet) 
    else:
        params_full['p'] = int(n_nets)
      
        params_full['is_trials_type'] = type_kernel
        full_A, full_phi, additional_return = run_GraFT(data = 'data_%s_xmin_0_xmax_n_ymin_0_ymax_n.npy'%type_grannet, corr_kern = [],  
              params = params_full , grannet=True, images = False, data_type = type_grannet , 
              labels_name = 'labels_%s_xmin_0_xmax_n_ymin_0_ymax_n.npy'%type_grannet) #'kernel_synth_xmin_0_xmax_70_ymin_0_ymax_70.npy''kernel_synth_grannet_xmin_0_xmax_n_ymin_0_ymax_n.npy'
            
       

                            
       
elif type_grannet ==  'trends_grannet':
    params_full['inverse_params'] = {'T_inverse':25}
    params_full['init_same'] =  np.random.choice([True,False])
    n_zer =  2 
    params_full['n_neighbors'] = np.random.randint(4,8)
    params_full['hard_thres_params'] = {'non_zeros': n_zer,
                                       'thres_error_hard_thres': 50,  'keep_last':False,
                                       'T_hard': 1}

    params_full['grannet_params']['rounded'] = True
    params_full['grannet_params']['rounded_max'] = 360
    
    params_full['is_trials'] = False
    params_full['phi_only_dec'] = False 
    params_full['noise_stuck_params'] = {'max_change_ratio': 0.01, 'in_a_row': np.random.randint(2,4), 
                                         'std_noise_A':0.25, 'std_noise_phi': 0.5, 'change_step': 20 }
    params_full['A_only_dec'] =False 
    type_kernel = 'shared' 

    params_full['is_trials_type'] = type_kernel
    params_full['epsilon'] = np.abs(np.random.randn()*0.06 + np.random.rand()*10 + 5)
    params_full['beta'] = np.abs(np.random.rand()*0.01 + 0.01)
    params_full['zeta'] =  np.random.rand()*20 + 30
    
    params_full['l1'] =  np.random.rand() + 1.4 # lambda 0
    params_full['solver'] = 'spgl1'
    
    params_full['weight_sim_nets'] =  np.random.rand()*10 + 20

    params_full['initial_thres'] =  np.random.choice([True,False])
    params_full['null_first'] = False
    params_full['null_late'] = np.random.choice([True,False])
    params_full['max_step_size'] = 30
    params_full['condition_unsupervised'] = True 
    params_full['phi_positive'] = True
    params_full['norm_A_cols'] =False # True
    params_full['l4'] = np.random.rand() + 0.45
    params_full['graph_params']['increase_sim'] = 1.01

    n_nets =  int(input('num nets?')   )
    
    if n_nets =='loop':
        rrr = str(np.random.randint(1,10000000))
        
        for j in range(3,15,3):
            params_full['name_addition']  = rrr + '_' + str(j)
            params_full['p'] = j
            params_full['max_learn'] = 300
            params_full['is_trials_type'] = type_kernel
            full_A, full_phi, additional_return = run_GraFT(data = 'data_%s_xmin_0_xmax_n_ymin_0_ymax_n.npy'%type_grannet, corr_kern = [],  
                  params = params_full , grannet=True, images = False, data_type = type_grannet , 
                  labels_name = 'labels_%s_xmin_0_xmax_n_ymin_0_ymax_n.npy'%type_grannet) #'kernel_synth_xmin_0_xmax_70_ymin_0_ymax_70.npy''kernel_synth_grannet_xmin_0_xmax_n_ymin_0_ymax_n.npy'
    else:
        params_full['p'] = int(n_nets)
        nu = np.ones(n_nets)
        params_full['is_trials_type'] = type_kernel
        full_A, full_phi, additional_return = run_GraFT(data = 'data_%s_xmin_0_xmax_n_ymin_0_ymax_n.npy'%type_grannet, corr_kern = [],  
              params = params_full , grannet=True, images = False, data_type = type_grannet , 
              labels_name = 'labels_%s_xmin_0_xmax_n_ymin_0_ymax_n.npy'%type_grannet, nu =nu) #'kernel_synth_xmin_0_xmax_70_ymin_0_ymax_70.npy''kernel_synth_grannet_xmin_0_xmax_n_ymin_0_ymax_n.npy'
            
                     

else:
    raise ValueError('unknown type_grannet')



def make_switch_state_data_for_plot(df_dict):
    res_list = []
    terms =  np.array(list(df_dict['CA'].columns)).astype(str)
    
    states_short = list(df_dict.keys())
    terms_full = np.repeat(terms, len(states_short))
    states_full = np.repeat(states_short,len(terms))
    for state in states_short :
        val = df_dict[state]
        for query_num, query in enumerate(terms):
            vals = val.loc[:,query].values
            res_list.append(vals.reshape((-1,1)))
    return np.hstack(res_list), states_full, terms
        
        
