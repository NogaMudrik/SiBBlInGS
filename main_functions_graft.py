"""
Created on Fri Sep  9 23:27:40 2022

@author: --

"""
"""
options for the kernel in grannet: "kernel_grannet_type"
    - "one_kernel"
    - "averaged" - average with others (then we need another param - 'params_weights')
    - "combination" - combination of shared kernel and individual kernels
    - "ind" - independent kernels
    need to change:
        - whenever mkDataGraph appear for grannet
        - lambda calculation

"""
# example: run_GraFT(data =[], corr_kern = [],  params = params_default, grannet=False)
#%% Imports
from sklearn.neighbors import NearestNeighbors
import os
import mat73
import scipy.io as sio
from sklearn.decomposition import PCA
from qpsolvers import solve_qp #https://scaron.info/doc/qpsolvers/quadratic-programming.html#qpsolvers.solve_qp https://pypi.org/project/qpsolvers/
#print(qpsolvers.available_solvers)
import matplotlib
#python3 -c 'from main_functions_graft import *; run_GraFT()'
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import seaborn as sns
import random
from datetime import date
import os.path
import warnings
from scipy.optimize import nnls
import numbers
from sklearn import linear_model
try:
    import pylops
except:
    print('did not load pylops')
#from PIL import Image
from skimage import io
global ask_selected
import networkx as nx
from datetime import datetime as datetime2

try:
    from pySPIRALTAP import *
    
    ask_selected = False # changehere True# str2bool(input('ask selected? (true or false)')) # True
    in_local = True #False
except:
    pwd_former = os.getcwd()
    os.chdir(r'E:/ALL_PHD_MATERIALS/pySPIRALTAP/pySPIRALTAP')
    import pySPIRALTAP
    from pySPIRALTAP import *
    os.chdir(pwd_former)
    ask_selected = False
    in_local = True
# #os.chdir(r'./pySPIRALTAP/pySPIRALTAP')
# os.chdir(r'pySPIRALTAP')

# import pySPIRALTAP
# from pySPIRALTAP import *
#from pySPIRALTAP import SPIRALTAP

#import pySPIRALTAP
#pwd_former = os.getcwd()
#os.chdir(r'./pySPIRALTAP/pySPIRALTAP')
#os.chdir(r'pySPIRALTAP')

#import pySPIRALTAP
#from pySPIRALTAP import *
#SPIRALTAP()
#os.chdir(pwd_former)
#from pySPIRALTAP import *
# except:    
#     pwd_former = os.getcwd()
#     os.chdir(r'E:/ALL_PHD_MATERIALS/pySPIRALTAP/pySPIRALTAP')
#     import pySPIRALTAP
#     from pySPIRALTAP import *
#     os.chdir(pwd_former)


def str2bool(str_to_change):
    """
    Transform 'true' or 'yes' to True boolean variable 
    Example:
        str2bool('true') - > True
    """
    if isinstance(str_to_change, str):
        str_to_change = (str_to_change.lower()  == 'true') or (str_to_change.lower()  == 'yes') or (str_to_change.lower()  == 't')
    return str_to_change

#%% Default Parameters 
global epsilon,params_default, instruct_per_selected
epsilon = 1e-5
sep = os.sep



instruct_per_selected = {'epsilon': 'Default tau values to be spatially varying, "tau" in MATLAB = 1'
                         ,'step_s': 'Default step to reduce the step size over time, 0.5' , 'p': 'Number of temporal profiles',
                         'nonneg':'Should be true or false',
                         'step_decay': 'should be around 0.99',
                         'reduceDim':'whether to apply pca before',
                         'solver': 'can be inv, spgl1, omp, IRLS, ista, fista, lasso, for solving Phi',
                         'norm_by_lambdas_vec':'Should be true to consider the weighted lasso',
                         'likely_from': 'poisson or gaussian',
                         'l1':'l1 regularization (rec around 0.01)', 
                         'l4': 'correlation between temporal activity'}





"""

important - to set the parameters choose type_answer = 1
"""
type_answer = 1#0 # 1 # can be random (0) or input (1)

"""
next steps
1) add a inverse steps sometimes 
2)  for synthetic - add an auto comparison to the real f and A
3)  display fig near the real one
4) choose the right step like in tSNE-pois

"""



##########################################################


if type_answer not in [0,1]: raise ValueError('undefined type_answer')

def decide_value(type_answer, text, val, type_choose = 'input'):
    if type_answer == 0:
        return val
    else:
        if type_choose == 'input':
            return input(text)
        else:
            return text

params_default = {'max_learn': 1e4, #1e3,  # Maximum number of steps in learning 
    'mean_square_error': 0.05,

    'deal_nonneg': 'make_nonneg', # can be 'make_nonneg' or 'GD'
    'epsilon' : decide_value(type_answer, 0.1, np.random.rand(), 'val'),                                                # Default tau values to be spatially varying
    'l1': decide_value(1, 0.1, np.random.rand(), 'val'),  #0.00007,#0.7,#0.7,                                             # Default lambda parameter is 0.6
    'l2': decide_value(type_answer, 0.1, np.random.rand(), 'val'), #0.00000000000005,# 0.2**2,    # lamForb                                      # Default Forbenius norm parameter is 0 (don't use)
    'l3': decide_value(type_answer, 0.1, np.random.rand(), 'val'),   #lamCont                 # Default Dictionary continuation term parameter is 0 (don't use)
    'lamContStp': decide_value(type_answer, 0.1, np.random.rand(), 'val')*0.9, #0.9,                                        # Default multiplicative change to continuation parameter is 1 (no change)
    'l4': decide_value(type_answer, 0.1, np.random.rand(), 'val')*0.1, #0.5,#0.1,     #lamCorr  # Default Dictionary correlation regularization parameter is 0 (don't use)
    'beta': 0.9*decide_value(type_answer, 0.1, np.random.rand(), 'val'),                                          # Default beta parameter to 0.09
    'maxiter': 0.01,                                            # Default the maximum iteration to whenever Delta(Dictionary)<0.01
    'numreps': 2,                                               # Default number of repetitions for RWL1 is 2
    'tolerance': 1e-7*decide_value(type_answer, 0.5, 0.8*np.random.rand(), 'val'),                                             # Default tolerance for TFOCS calls is 1e-8

    'likely_from' : decide_value(type_answer, 'gaussian' , 'gaussian', 'val') ,# , CHANGEHERE THE 1ST INDEX 'likely from poisson/gaussian'      poisson                   # Default to a gaussian likelihood ('gaussian' or'poisson')
    'step_s': decide_value(type_answer, 0.2, 0.1+0.9*np.random.rand(), 'val'), #5,1,                                               # Default step to reduce the step size over time 
                                                                        # (only needed for grad_type ='norm')
    'step_decay': np.min([0.9999,0.995 + 0.005*decide_value(type_answer, 0.9, np.random.rand(), 'val')]),                                           # Default step size decay (only needed for grad_type ='norm')
                                       
    'dict_max_error': 0.01,       # learn_eps                                    # Default learning tolerance: stop when Delta(Dictionary)<0.01
    'p': 7,                        # Default number of dictionary elements is a function of the data
    'verb': 1,                                               # Default to no verbose output
  
    'GD_iters': 1* decide_value(type_answer, 1, np.random.randint(1,5), 'val'),                                               # Default to one GD step per iteration
    'bshow': 0,                                               # Default to no plotting
                                                  # Default to not having negativity constraints
    'nonneg': decide_value(1, False, False,'val')   ,  # CHANGEHERE!!! TO 'nonneg True/False, False for voltgraft, True for grannet'                           # Default to not having negativity constraints on the coefficients
    'plot': False,                                           # Default to not plot spatial components during the learning
    'updateEmbed' : False,                                           # Default to not updateing the graph embedding based on changes to the coefficients
    'mask': [],                                              # for masked images (widefield data)
    'normalizeSpatial' : False,                                      # default behavior - time-traces are unit norm. when true, spatial maps normalized to max one and time-traces are not normalized     
     'patchSize': 50, 
     'motion_correct': False, #future
     'kernelType': 'embedding',
     'reduceDim': decide_value(type_answer, False, np.random.choice([False, True]), 'val'),    
     'w_time': 0,
     'n_neighbors': decide_value(type_answer,49, np.random.randint(5,50), 'val'),    
     'n_comps':5,
     'solver_qp':'quadprog',
     'solver': decide_value(type_answer,'inv', np.random.choice(['spgl1','inv', 'lasso']), 'val'),   
     'nullify_some': False , 
     'norm_by_lambdas_vec': decide_value(1,False, np.random.choice([False, True]), 'val'),  
     'min_max_data': False,
     'GD_type': 'full_ls_cor', # 'norm',#
     'multi':'med',  # can be med, sqrt, none
     'thresANullify':-50, #0,
     'CI': {
     'xmin' : 151,#151
     'xmax' : 200,#350                                                            # Can sub-select a portion of the full FOV to test on a small section before running on the full dataset
     'ymin' : 151,#101
     'ymax' : 200,#300 
     },
     'VI_crop': {
     'xmin' : 120,#151
     'xmax' : 270,#350                                                            # Can sub-select a portion of the full FOV to test on a small section before running on the full dataset
     'ymin' : 0,#101
     'ymax' : -1,#300 
     },
     'VI_crop_very': {
     'xmin' : 120,#151
     'xmax' : 170,#350                                                            # Can sub-select a portion of the full FOV to test on a small section before running on the full dataset
     'ymin' : 20,#101
     'ymax' : 70,#300 
     },     
     'VI_full': {
     'xmin' : 0,#151
     'xmax' : -1,#350                                                            # Can sub-select a portion of the full FOV to test on a small section before running on the full dataset
     'ymin' : 0,#101
     'ymax' : -1,#300 
     },
     'area2': {
     'xmin' : 0,#151
     'xmax' : 'n',#350                                                            # Can sub-select a portion of the full FOV to test on a small section before running on the full dataset
     'ymin' : 0,#101
     'ymax' : 'n',#300 
     },
     'trends': {
     'xmin' : 0,#151
     'xmax' : 'n',#350                                                            # Can sub-select a portion of the full FOV to test on a small section before running on the full dataset
     'ymin' : 0,#101
     'ymax' : 'n',#300 
     },     
     'VI_HPC': {
     'xmin' : 0,#151
     'xmax' : 'n',#350                                                            # Can sub-select a portion of the full FOV to test on a small section before running on the full dataset
     'ymin' : 0,#101
     'ymax' : 'n',#300 
     },       
     'VI_crop_long': {
     'xmin' : 0,#151
     'xmax' : 128,#350                                                            # Can sub-select a portion of the full FOV to test on a small section before running on the full dataset
     'ymin' : 200,#101
     'ymax' : 328,#300 
     },     
     'VI_crop_long2': {
     'xmin' : 20,#151
     'xmax' : 100,#350                                                            # Can sub-select a portion of the full FOV to test on a small section before running on the full dataset
     'ymin' : 200,#101
     'ymax' : 280,#300 
     },  
    'synth': {
    'xmin' : 0,#151
    'xmax' : 70,#350                                                            # Can sub-select a portion of the full FOV to test on a small section before running on the full dataset
    'ymin' : 0,#101
    'ymax' : 70,#300 
     },   
    'synth_grannet':
        {
    'xmin' : 0,#151
    'xmax' : 'n',#350                                                            # Can sub-select a portion of the full FOV to test on a small section before running on the full dataset
    'ymin' : 0,#101
    'ymax' : 'n',#300 
            },
    'synth_trials_grannet':
        {
    'xmin' : 0,#151
    'xmax' : 'n',#350                                                            # Can sub-select a portion of the full FOV to test on a small section before running on the full dataset
    'ymin' : 0,#101
    'ymax' : 'n',#300 
            },
    'trends_grannet' :
    {
    'xmin' : 0,#151
    'xmax' : 'n',#350                                                            # Can sub-select a portion of the full FOV to test on a small section before running on the full dataset
    'ymin' : 0,#101
    'ymax' : 'n',#300 
            },
    'neuro_bump_short'  :
    {
    'xmin' : 0,#151
    'xmax' : 'n',#350                                                            # Can sub-select a portion of the full FOV to test on a small section before running on the full dataset
    'ymin' : 0,#101
    'ymax' : 'n',#300 
            },      
    'neuro_bump_short_short'  :
    {
    'xmin' : 0,#151
    'xmax' : 'n',#350                                                            # Can sub-select a portion of the full FOV to test on a small section before running on the full dataset
    'ymin' : 0,#101
    'ymax' : 'n',#300 
            },   
    'neuro_bump_angle_active' :
    {
    'xmin' : 0,#151
    'xmax' : 'n',#350                                                            # Can sub-select a portion of the full FOV to test on a small section before running on the full dataset
    'ymin' : 0,#101
    'ymax' : 'n',#300 
            }, 
    'VI_Peni':{
        
        'xmin' : 900,#151
        'xmax' : 1000,#350                                                            # Can sub-select a portion of the full FOV to test on a small section before running on the full dataset
        'ymin' : 900,#101
        'ymax' : 1000,#300 
        },
    'VI_Peni_partt':{
        
        'xmin' : 930,#151
        'xmax' : 1000-10,#350                                                            # Can sub-select a portion of the full FOV to test on a small section before running on the full dataset
        'ymin' : 930,#101
        'ymax' : 1000-10,#300 
        },
     'use_former_kernel' : False,
     'usePatch' : False, #str2bool(input('to use patch? (true or false)') )
     'portion' :True,# str2bool(input('To take portion?'))
     'divide_med' : False,# str2bool(input('divide by median? (true or false)'))
     'data_0_1' : False,
     'to_save' : True, #str2bool(input('to save?'))
     'default_path':  r'E:/CODES FROM GITHUB/GraFT-analysis/code/neurofinder.02.00/images', 
     'save_error_iterations': True,
     'max_images':800,
     'dist_init': 'randn', #'uniform' # 'rand', # Can also be rand, uniform or randn
     'to_sqrt':True,
     'Poisson':{'maxiter': 5, 'miniter':0.01, 'stopcriterion': 3,
                'tolerance': 1e-8, 'alphainit': 1, 'alphamin': 1e-30, 
                'alphamax': 1e30, 'alphaaccept': 1e30, 'logepsilon': 1e-10,
                'saveobjective':True, 'savereconerror':False, 'savecputime':True, 'penalty':'Canonical',
                'savesolutionpath':False, 'truth': False,'alphamethod':1, 'monotone' : 1, },  # try to change to 0,0  the 2 last ones
     'sigma_noise': 0.1, # std of noise to add to dict if all values are zeros
     'grannet_params':{'lambda_grannet': 0.1, 
                       'distance_metric':'Euclidean',
                       'reg_different_nets': 'unified',
                       'num_free_nets':0,
                       'distance2proximity_trans': 'exp',
                       'initialize_by_other_nets':True, # True if to initialize A by the other nets
                       'late_start': 5,
                       'include_Ai':False,  # whether to include the distance from Ai to itself in NeuroInfer (if True - kind of smoothness regularization)
                       'labels_indicative':False },
     'to_store_lambdas':False,
     'reorder_nets_by_importance': False,
     'uniform_vals': [0,1],
     'distance_metric': 'euclidean' ,
     'save_plots': True,
      'graph_params':{'kernel_grannet_type':'combination', 'params_weights': 0.4} ,# params_weights should be a vector if kernel_grannet_type is averaged and a scalar if combined
      'hard_thres' : True, # if to take hard thres on A
      'hard_thres_params': {'non_zeros': 21, 'thres_error_hard_thres': 3, 'T_hard': 1},
      'add_inverse': True,
      'inverse_params': {'T_inverse':2},
      'compare_to_synth': True,
      'various_steps': True, # if to check multiple steps for GD
      'steps_range': np.arange(0.5, 1.5, 0.05), #,np.array([0.7,0.9, 1,1/0.7, 1/0.9, 0.3,3]), 
      'step_loc': 1,
      'phi_only_dec':True,
      'A_only_dec':False,
      'min_add_phi_error': 0.05, # 0.05 of current error
      'T_save_fig': 40, 
      'is_trials': False,
      'is_trials_type': 'shared',   # is trials type can be shared = shared kernel, shared ners / shared_flex - shared kernel, flexible nets / flex - different kernels, different nets
      'num_nets_sample_trials': 0 , # "apply in the future!!", # how many networks to consider in a batch?
      'add_noise_stuck':  True     ,     # 
      'noise_stuck_params': {'max_change_ratio': 0.05, 'in_a_row': 5, 'std_noise_A':0.05, 'std_noise_phi': 0.3, 'change_step': 20 },  # in_a_row = how many times  the error remains; max_change_ratio - ratio of error that is considered unchanged
      'condition_supervised': False,
      'to_norm_cond_graph': True,
      'min_step_size': 1e-10, 
      'max_step_size': 50, 
      'addition_norm_cond':1 # how much to add to the normalization of similar nets
      
    }
if  params_default['likely_from'].lower() not in ['gaussian','poisson']:
    raise ValueError('likely_from must be poisson or gaussian')


"""
check params
"""
if params_default['graph_params']['kernel_grannet_type'] not in ["one_kernel", "averaged", "combination", "ind" ]:
    raise ValueError('invalid kernel_grannet_type in graph_params (current %s)'%params_default['graph_params']['kernel_grannet_type'])
    

if params_default['is_trials'] and params_default['is_trials_type'] not in ['shared' , 'shared_flex', 'flex']:
    raise ValueError("invalid is_trials_type, shoulbe be in ['shared' , 'shared_flex', 'flex'] but %s"%params_default['is_trials_type'] )




global data_types
data_types  = ['CI', 'VI_crop', 'VI_full', 'VI_crop_very', 'area2', 'trends', 'VI_HPC', 'synth_trials_grannet','VI_Peni_partt',
               'VI_crop_long', 'VI_crop_long2','synth','synth_grannet','trends_grannet','neuro_bump_short','neuro_bump_short_short', 'neuro_bump_angle_active',
               'VI_Peni']


#%%  GraFT Functions

global params_config
params_config = {'self_tune':7, 'dist_type': 'euclidian', 'alg':'ball_tree',
                       'n_neighbors':49, 'reduce_dim':False}

    
    
def check_error_stuck(last_errors, params_stuck)    :
    """
    Checks if the algorithm is stuck or not by analyzing the change in error.
    
    Args:
        last_errors (list): A list of the previous errors in descending order.
        A (numpy.ndarray): A matrix used in the algorithm.
        phi (numpy.ndarray): A vector used in the algorithm.
        params_stuck (dict): A dictionary containing parameters related to checking if the algorithm is stuck.
    
    Returns:
        bool: True if the algorithm is stuck, False otherwise.
    """
    # if return True -> stuck
    # if return false - >not stuck but decrease
    if len(last_errors) > params_stuck['in_a_row']:
        last_errors =  last_errors[-params_stuck['in_a_row']:]
    max_thres = params_stuck['max_change_ratio']*last_errors[-1]
    return (np.abs(np.diff(last_errors)) < max_thres).all()
    
def add_noise_if_stuck(last_errors, A, phi, step_GD,  params_stuck) :
    if check_error_stuck(last_errors, params_stuck) :
        print('add noise @@@@@@@@@@@@@ !!!!!!!!!!!!!! @@@@@@@@@@@@@@@@')
        ss = int(str(datetime2.now()).split('.')[-1])
        np.random.seed(ss)
        A = A + np.random.randn(*A.shape)*params_stuck['std_noise_A']
        phi = phi + np.random.randn(*phi.shape)*params_stuck['std_noise_A']
        step_GD *= params_stuck['change_step']
    return A,phi, step_GD
  


def labels_to_nums(labels):
    dict_nums_labels ={}
    list_nums = []
    for label_num, label in enumerate(labels):
        dict_nums_labels[label_num] = label
        list_nums.append(label_num)
    return dict_nums_labels, list_nums
    
def labels2proximity(labels, distance_metric = 'Euclidean', distance2proximity_trans = 'exp'):
    """  
    Parameters
    ----------
    labels : TYPE
        DESCRIPTION.

    Returns
    -------
    distance : np.array of #labels X #labels (#trials X #trials)
        i-th row - what is the distance to the i-th label
        j-th col - the labels count
    proximity :  np.array of #labels X #labels (#trials X #trials)
        i-th row - what is the Proximity to the i-th label
        j-th col - the labels count
    """
    distance_base = np.repeat(labels.reshape((1,-1)), len(labels), axis = 0) - labels.reshape((-1,1))
    if distance_metric == 'Euclidean':
        distance = distance_base ** 2
    elif distance_metric == 'abs':
        distance = np.abs(distance_metric)       
    else:
        raise ValueError('Unknown Distance Metric!')
    if distance2proximity_trans == 'exp':
        proximity = np.exp(-distance)
    elif  distance2proximity_trans == 'inv':
        proximity = 1/distance
    else:
        raise ValueError('Unknown Proximity Metric!')     
    
    return distance, proximity


def lists2list(xss)    :
    return [x for xs in xss for x in xs] 

def create_proximity_coeffs_based_on_prxoimity(p, proximity = [], num_free_nets = 0, reg_different_nets = 'unified',
                                               num_states = 0, nu = [], params = params_default, data = []):
    """
    This function gets a proximity matrix between the LABELS of different trials and is num_trials X num_trials
    if emptylist -> (in case the label is unknown, should not be considered or the same) - form the same proximity to all states
    
    
    Parameters
    ----------
    p: how many nets?
    proximity : np.array of #labels X #labels 
        DESCRIPTION.
    num_free_nets : scalar 
        if integer - number of free (flexible) nets
        if fraction - percent of free (flexible) nets
    num_states: integer >= 0
        needed only in case proximity is an empty list. Indicates the full number of trials
    nu: giving different weights to different nets
        
    Returns
    -------
    coeffs : np.array [# trials X # trials X # nets] (for the synth data it is 3 X 3 X 10)
        1. Which trial we are currently optimizing?
        2. From which trial do we calculate the distance from?
        3. Which network we are talking about?
    (will later be used in graft_with_grannet for coefficients_similar_nets)

    """
    
    if params['condition_supervised'] and checkEmptyList(data):
        raise ValueError('data should not be emptylist if "condition_supervised"')
    elif checkEmptyList(proximity) and params['condition_supervised'] :
        proximity  = cal_dist(0, data, graph_params = params['graph_params'],
                 grannet = True, distance_metric = params['distance_metric'])
        if params['to_norm_cond_graph']:
            proximity = norm_vec_min_max(proximity) + params['addition_norm_cond']
        
    elif checkEmptyList(proximity):
        proximity = np.ones((num_states, num_states, 1))
        
   
    if not checkEmptyList(nu):
        if len(nu) != p:
            raise ValueError('Len of $nu$ (%d) must be equal to the number of nets (%d)'%(len(nu), p))
        coeffs_mat = np.dstack([proximity*nu_i for nu_i in nu])
        #np.repeat(proximity.reshape((proximity.shape[0], proximity.shape[1],1)), p, axis = 2)
    else:
        if reg_different_nets == 'unified': # the kernel is the same
            coeffs_mat = np.repeat(proximity.reshape((proximity.shape[0], proximity.shape[1],1)), p, axis = 2)
        elif reg_different_nets == 'random':
            coeffs_mat= np.random.rand(proximity.shape[0], proximity.shape[1], p)
        
        elif reg_different_nets == 'decreasing_exp' :
            p_base = np.arange(p).reshape((1,1,-1))
            p_base = np.exp(-p_base)
            p_base = p_base/np.max(p_base)
            p_base = np.repeat(p_base, proximity.shape[0], axis = 0)
            p_base = np.repeat(p_base, proximity.shape[1], axis = 1)
            coeffs_mat = p_base*proximity.reshape((proximity.shape[0], proximity.shape[1],1))
            if not p_base.shape == (proximity.shape[0], proximity.shape[1], p):
                raise IndentationError('Invalid shape for p_base for decreasing_exp')
            #coeffs = np.random.rand(proximity.shape[0], proximity.shape[1], p)
        elif reg_different_nets == 'decreasing_step' :
            p_base = np.arange(p).reshape((1,1,-1))[::-1]
            p_base = p_base/np.max(p_base)
            p_base = np.repeat(p_base, proximity.shape[0], axis = 0)
            p_base = np.repeat(p_base, proximity.shape[1], axis = 1)
            coeffs_mat = p_base*proximity.reshape((proximity.shape[0], proximity.shape[1],1))
            if not p_base.shape == (proximity.shape[0], proximity.shape[1], p):
                raise IndentationError('Invalid shape for p_base for decreasing_step')
                
        else:
            raise ValueError('Unknown reg_different_nets')
    coeffs_mat[:,:,-num_free_nets:] = 0
    
    return coeffs_mat



def createDefaultParams(params = {}):
    dictionaryVals = {'step_s':1, 
                      'learn_eps':0.01,
                      'epsilon': 2,
                      'numreps': 2, 
                      }
    return  addKeyToDict(dictionaryVals,params)

def createLmabdasMat(epsilonVal, shapeMat):
    if isinstance(epsilonVal,  (list, tuple, np.ndarray)) and len(epsilonVal) == 1:
        epsilonVal = epsilonVal[0]
    if not isinstance(epsilonVal, (list, tuple, np.ndarray)):
        labmdas = epsilonVal * np.ones(shapeMat)
    else:
        epsilonVal = np.array(epsilonVal)
        if len(epsilonVal) == shapeMat[1]:
            lambdas = np.ones(shapeMat[0]).rehspae((-1,1)) @ epsilonVal.reshape((1,-1))
        elif len(epsilonVal) == shapeMat[0]:
            lambdas =  epsilonVal.reshape((-1,1)) @  np.ones(shapeMat[1]).rehspae((1,-1))
        else:
            raise ValueError('epsilonVal must be either a number or a list/tupe/np.array with the a number of elements equal to one of the shapeMat dimensions')






def addKeyToDict(dictionaryVals,dictionaryPut):
    return {**dictionaryVals, **dictionaryPut}


def validate_inputs(params):
    """
    This function takes a dictionary of parameters as input and validates them.
    Parameters
    ----------
    params : dict;         A dictionary of parameters to be validated.
    
    Returns
    -------
    dict;         A validated dictionary of parameters.
    """    
    params['epsilon'] = float(params['epsilon'])
    params['step_s'] = float(params['step_s'])
    params['l1'] = float(params['l1'])
    params['l4'] = float(params['l4'])
    params['p'] = int(params['p'])
    params['nonneg'] = str2bool(params['nonneg'])
    params['reduceDim'] = str2bool(params['reduceDim'])

        
    params['solver'] = str(params['solver'])
    params['step_decay'] = float(params['step_decay'])
    params['norm_by_lambdas_vec'] = str2bool(params['norm_by_lambdas_vec'])
    params['likely_from'] = str(params['likely_from'])
    return params

def plot_nets_side_by_size(A1,A2, real_axis = 1, ax = [], linewidth = None, linecolor = None,cmap = None, cbar = None):
    #print(A1.shape); print(A2.shape)
    # real axis is the num of the nets
    if checkEmptyList(ax): fig, ax = plt.subplots()
    conc = []
    if real_axis == 1:
        #A1_A2 = np.hstack([ np.hstack([(A1[:,p]/np.max(A1[:,p])).reshape((-1,1)), (A2[:,p]/np.max(A2[:,p])).reshape((-1,1))])  for p in range(A1.shape[1])])
        A1_A2 = np.hstack([np.hstack([ norm_vec_min_max(A1[:,p]).reshape((-1,1)), norm_vec_min_max(A2[:,p]).reshape((-1,1))])  for p in range(A1.shape[1])])
    else:
        A1_A2 = np.vstack([np.vstack([ norm_vec_min_max(A1[p,:]).reshape((1,-1)), norm_vec_min_max(A2[p,:]).reshape((1,-1))])  for p in range(A1.shape[0])])
        #A1_A2 = np.vstack([ np.vstack([A1[p,:]/np.max(A1[p,:]) , A2[p,:]/np.max(A2[p,:])   ])  for p in range(A1.shape[0])])
    sns.heatmap(A1_A2, ax = ax, robust = True, 
                
                linewidth = linewidth, linecolor = linecolor,cmap =cmap , cbar = cbar )
        
        
def load_data(data_type = [],  xmin = '0', xmax = 'n',ymin = '0',ymax = 'n', params = params_default, 
              type_name = 'data', type_answer = type_answer):
    """
    Load the data according to the data type and specified x and y limits.
    
    Parameters
    ----------
    data_type : list or str, optional
        List of data types to be loaded, by default []
    xmin : str, optional
        Minimum limit for x-axis, by default '0'
    xmax : str, optional
        Maximum limit for x-axis, by default 'n'
    ymin : str, optional
        Minimum limit for y-axis, by default '0'
    ymax : str, optional
        Maximum limit for y-axis, by default 'n'
    params : dict, optional
        Dictionary containing default parameters, by default params_default
    type_name : str, optional
        String representing the type of data to be loaded, by default 'data'
        
    Returns
    -------
    tuple
        A tuple consisting of the data type and the name of the data file
        
    Raises
    ------
    ValueError
        If the data type is not recognized
    ValueError
        If the data type is not a string
        
    """    
    if checkEmptyList(data_type):
        if not in_local:
            data_type_ask = decide_value(type_answer, 'short or long? (answer short or long)', 'long')
            
            if data_type_ask.lower() == 'short':
                data_type = 'VI_crop_very'
            else:
                data_type = 'VI_full'
        else:
            data_type =  decide_value(type_answer, 'data type? (can be CI for calcium-imaging, VI for voltage imaging), synth for synthetic', 'synth')
            #input)

        if data_type not in data_types:
            raise ValueError('Unknown Data Type (if you think it is a mistake, please add this data to "data_types")')
        xmin = params[data_type]['xmin']; xmax = params[data_type]['xmax']
        ymin = params[data_type]['ymin']; ymax = params[data_type]['ymax']
       
    if isinstance(data_type, str):
        return data_type, create_data_name(data_type, xmin, xmax, ymin, ymax, type_name)
    else:
        raise ValueError('datatype should be string in load_data')
        
def create_data_name(data_name = '', xmin = '0', xmax = 'n',ymin = '0',ymax = 'n', type_name = 'data'):
    """
    This function creates a string with a specified format for data file names.
    
    Parameters:
    data_name (str, optional): The name of the data. Default value is an empty string.
    xmin (str, optional): The lower limit of the x axis. Default value is '0'.
    xmax (str, optional): The upper limit of the x axis. Default value is 'n'.
    ymin (str, optional): The lower limit of the y axis. Default value is '0'.
    ymax (str, optional): The upper limit of the y axis. Default value is 'n'.
    type_name (str, optional): The type of data. Default value is 'data'.
    
    Returns:
    str: The generated string in the format 'type_name_data_name_xmin_xmax_ymin_ymax.npy'.
    
    Example:
    >>> create_data_name('data_sample', '-5', '5', '-10', '10', 'experiment')
    'experiment_data_sample_xmin_-5_xmax_5_ymin_-10_ymax_10.npy'
    """    
    return '%s_%s_xmin_%s_xmax_%s_ymin_%s_ymax_%s.npy'%(type_name, data_name,str(xmin), str(xmax), str(ymin), str(ymax))
    

def split_stacked_data(data, T = 0, k = 0):
    # data in shape Neurons X (Time X conditions) [such that [bluck 1 N X T] , bluck 2 N X T]
    # PAY ATTNETION !! RETURN N x TIME x CONDITION (if applying for phi in grannet - transpose axis 0,1)
    if len(data.shape) == 3:
        return np.dstack([split_stacked_data(data[:,:,k_global], T, k) for k_global in range(data.shape[2])] )
    if T == 0 and k == 0:
        raise ValueError('you must provide either k and T!')
    elif T != 0 and k!= 0:
        if k*T != data.shape[1]:
            print('k')
            print('T')
            print(k)
            print(T)
            raise ValueError('T*k must be equal to 2nd dim of data (data.shape[2]), but t*k = %d and data.shape[1] = %d'%(T*k, data.shape[1]))
        else:
            pass
    elif T == 0:
        T = data.shape[1] / k
    elif k == 0:
        k = data.shape[1]/T
    else:
        raise ValueError('how did you arrive here?')
    if k != int(k) or T != int(T):
        raise ValueError('T and k must be ints. but T and k are %s'%str((T,k)))
    else:
        k = int(k)
        T = int(T)
    groups_opts = np.linspace(0, k*T, k+ 1).astype(int)
    print(groups_opts)
    store_data = []
    for opt_count, opt_begin in enumerate(groups_opts[:-1]):
        opt_end = groups_opts[opt_count + 1]
        cur_data = data[:,opt_begin : opt_end ]
        store_data.append(cur_data)
    return np.dstack(store_data)
        
        
    
    
    
    
def run_GraFT(data = [], corr_kern = [], params = {}, to_save = True, to_return = True,
               ask_selected = ask_selected, selected = ['epsilon','step_s', 'p', 'nonneg','step_decay'
                                               ,'solver', 'l4', 'l1'], grannet = False,
               label_different_trials = [], save_mid_results = True, type_answer = type_answer,
               instruct_per_select = instruct_per_selected, nu = [], images = False, data_type = '', path_name = '',
               labels_name = '', labels = []):
    """
    This function runs the main graft algorithm.

    Parameters
    ----------
    data : can be string of data path or a numpy array (pixels X pixels X time) or (pixels X time). 
        Leave empty for default
        The default is []. In this case the calcium imaging dataset will be used. 
        if grannet - 
        
    corr_kern : proximity kernel. Leave empty ([]) to re-create the kernel.  
    params : dictionary of parameters, optional
        the full default values of the optional parameters are mentioned in dict_default. 
    to_save : boolean, optional
        whether to save the results to .npy file. The default is True.
    to_return : boolean, optional
        whether to return results. The default is True.
    ask_selected : boolean, optional
        whether to ask the use about specific parameters. The default is True.
    selected : list of strings, optional
        relevant only if 'ask_selected' is true. 
        The default is ['epsilon','step_s', 'p', 'nonneg', 'reduceDim','solver','norm_by_lambdas_vec'].
    nu: list/np.array, describing how much weight to give to different nets. Its length must be equal to p (or empty)
    images: bool. If true - assume pixels X pixels X time
      
    save_mid_results: boolean; whether to save intermediate results

    Raises
    ------
    ValueError
        If invalid path

    Returns
    -------
    A : np.ndarray (pixels X p) - neural maps
    phi : np.ndarray (time X p)   temporal traces
    additional_return : dictionary with additional returns. Including error over iterations
    """    
    params['data_name'] = data
    params = {**params_default, **params}
    
    params['compare_to_synth'] = params['compare_to_synth'] and 'synth_grannet' in data and grannet
    

    
    if ask_selected:
        for select in selected:
            params[select] = input('Value of %s (def = %s) \n %s'%(select, str(params[select]), instruct_per_selected[select]))
            params = validate_inputs(params)
    #selected_name = '_'.join([val + '-' + params[val] for val in selected] ) + '.npy'
    if to_save:
        save_name =  str(datetime2.now()).replace(':','_').replace('.','_').replace('-','_')+'_' + str(np.random.randint(0,10)) + '_g'
        #decide_value(type_answer,'save_name',                                   ,  'input') # changehere
        if ask_selected:
            addition_name = '_'.join([s +'-' + str(params[s] ) for s in selected])
            save_name_short = save_name
            save_name = save_name + '_' + addition_name
        print(save_name)
     
    else:
        save_name = 'def'
    #print(save_name)
 
    """
    create path name
    """
    if len(path_name) ==0:
        path_name = data_type + os.sep + str(date.today()) + os.sep 
        
    """
    Create data
    """
    default = False
    if checkEmptyList(data):
               
        default = True
        
        data_type, data = load_data(data_type = [])
       
        if checkEmptyList(label_different_trials) and grannet:
            _, label_different_trials = load_data(data_type = data_type, type_name='labels', **params[data_type] )
    elif isinstance(data, str) and len(data_type) == 0:
        if data.endswith('.npy'):
            data_type = data.split('_')[1]
        else:
            data_type = data
        
    if isinstance(data, str): # Check if path
        been_cut = False
        if grannet:
            data = np.load(data)  
            been_cut = True
            if checkEmptyList(label_different_trials):
                name_label = create_data_name(data_name = data_type,  type_name = 'labels', **params[data_type]) 

                label_different_trials =  np.load(name_label)
                
                
            
        else:
            try:
                try:
                    if data.endswith('.npy'):
                        #data = np.load('data_calcium_xmin_%d_xmax_%d_ymin_%d_ymax_%d.npy'%(params['xmin'],params['xmax'],params['ymin'],params['ymax'])) 
                        data = np.load(data)  
                        been_cut = True
                        if data_type == 'trends' and data.shape[0] > data.shape[1]:
                            data = data.T
                    else:
                        data =  from_folder_to_array(data, max_images = params['max_images'])   
                except:
                    if default:
                        data =  from_folder_to_array(params['default_path'], max_images = params['max_images'])  
                    else:
                        raise ValueError('Data loading failed')
                    
            except:
                raise ValueError('Cannot locate data path! (your invalid path is %s)'%data)
    
    """
    check if trials - call labels
    """
    if params['is_trials'] and checkEmptyList(labels):
        if len(labels_name) == 0:
            labels_name = create_data_name(data_name = data_type,  type_name = 'labels')
        try:
            labels = np.load(labels_name)
            lab_unique, trials_counts = np.unique(labels, return_counts = True)      
            
        except:
            raise ValueError('unable to load labels, try %s'%labels_name)        
        params['trials_counts'] = trials_counts
        params['lab_unique'] = lab_unique
        if params['is_trials_type'] == 'shared_flex': # same kernel, different nets
            #herehere change kernel
            data_full = data.copy()            
            # make data for each label
            # each el will be neurons X Time X trials for k_spec
            data = np.dstack([MovToD2_grannet(data[:,:,labels == k]) for  k in lab_unique])
            
        elif params['is_trials_type'] == 'flex': 
            pass
        elif params['is_trials_type'] == 'shared': 
            """
            in this case - the kernel witl be only k d (no trials included) and A will also be only k d. however
            Y and phi will be kXtirals dimensions
            """
            data_full = data.copy()            
            # make data for each label
            # each el will be neurons X Time X trials for k_spec
            data = np.dstack([MovToD2_grannet(data[:,:,labels == k]) for  k in lab_unique])
            
            
        else:
            raise ValueError("how you arrive here? params_default['is_trials_type'] %s"%params['is_trials_type'])
    
    
    
    
    """
    define save path
    """
    if to_save:
        path_name = data_type + os.sep + str(date.today()) 
        path_name = path_name ='results_march_2023' + os.sep + data_type + os.sep + str(date.today()) + os.sep + save_name + '_folder' 
        #
        
        
    if images:
        data = MovToD2(data)
        
    if (is_2d(data, 3) and not grannet) or (is_2d(data,2) and grannet):
        raise ValueError('If grannet - data should be 3d; If not grannet - data should be 2d')                
    if isinstance(data,np.ndarray):
        if not been_cut:
            data = data[params[data_type]['xmin']:params[data_type]['xmax'], 
                        params[data_type]['ymin']:params[data_type]['ymax'],:]
            np.save('data_%s_xmin_%d_xmax_%d_ymin_%d_ymax_%d.npy'%(data_type,params[data_type]['xmin'],
                                                                   params[data_type]['xmax'],
                                                                   params[data_type]['ymin'],
                                                                   params[data_type]['ymax']), 
                    data)  
    if params['likely_from'].lower() == 'poisson':
        data = np.round(data).astype(int)
    if params['min_max_data']:
        data = (data - np.min(data, 1).reshape((-1,1))) / (data.max(1) - data.min(1)).reshape((-1,1))

    
    """
    create grannet labels 
    """

    if grannet:
        if  params['condition_supervised']:
            _, proximity = labels2proximity(label_different_trials, 
                                            distance_metric = params['grannet_params']['distance_metric'], 
                                            distance2proximity_trans = params['grannet_params']['distance2proximity_trans'])
            num_states = proximity.shape[0]
        else:
            num_states = data.shape[2]
            proximity = []
        if params['is_trials'] and params['is_trials_type'] == 'shared' and not params['condition_supervised']:
            if data.shape[2] != len(np.unique(labels)):
                data = np.dstack([MovToD2_grannet(data[:,:,labels == k]) for  k in lab_unique])
            coefficients_similar_nets = create_proximity_coeffs_based_on_prxoimity(params['p'], proximity, num_free_nets = params['grannet_params']['num_free_nets'], 
                                                                                   reg_different_nets = params['grannet_params']['reg_different_nets'],
                                                                                   num_states = len(np.unique(labels)), nu = nu, params = params, data = data) 

        else:
            coefficients_similar_nets = create_proximity_coeffs_based_on_prxoimity(params['p'], proximity, num_free_nets = params['grannet_params']['num_free_nets'], 
                                                                                   reg_different_nets = params['grannet_params']['reg_different_nets'],
                                                                                   num_states = num_states, nu = nu, params = params, data = data)        
    """
    Create Kernel
    """
    print('creating kernel...')
    """
    grannet case
    """
    if grannet:
        if checkEmptyList(corr_kern):
            use_former = params['use_former_kernel']
            """
           
            depending on the graph type - decide how to create the kernel
            
            if independent kernels:
            """
            if params['graph_params']['kernel_grannet_type'] == 'ind':
                corr_kern = np.dstack([mkDataGraph(data[:,:,trial_counter], params, reduceDim = params['reduceDim'], 
                                     reduceDimParams = {'alg':'PCA'}, graph_function = 'gaussian',
                            K_sym  = True, use_former = use_former, data_name = data_type,
                            toNormRows = True,  graph_params = params['graph_params'], grannet = grannet) for trial_counter in range(data.shape[2])])
            
            elif params['graph_params']['kernel_grannet_type'] == 'one_kernel':
                # THIS OPTION FIND ONE KERNEL SHARED AMONG ALL CONDITIONS
                
                shared_kernel = mkDataGraph( MovToD2_grannet(data), params, reduceDim = params['reduceDim'], 
                                     reduceDimParams = {'alg':'PCA'}, graph_function = 'gaussian',
                            K_sym  = True, use_former = use_former, data_name = data_type,
                            toNormRows = True,  graph_params = graph_params, grannet = grannet)
                corr_ker = np.dstack([shared_kernel]*data.shape[2])
                # corr_kern = np.dstack([mkDataGraph(data[:,:,trial_counter], params, reduceDim = params['reduceDim'], 
                #                      reduceDimParams = {'alg':'PCA'}, graph_function = 'gaussian',
                #             K_sym  = True, use_former = use_former, data_name = data_type, toNormRows = True, data = data, graph_params = params['graph_params'], grannet = grannet) for trial_counter in range(data.shape[2])])
            
            elif params['graph_params']['kernel_grannet_type'] in ['combination',  'averaged']:
                # THIS OPTION USE A COMBINATION OF SHARED KERNEL AND INDIVIDUAL KERNEL
          
                corr_kern = mkDataGraph_grannet(data, params, reduceDim = params['reduceDim'], 
                                     reduceDimParams = {'alg':'PCA'}, graph_function = 'gaussian',
                            K_sym  = True, use_former = use_former, data_name = data_type,
                            toNormRows = True,  graph_params = params['graph_params'], 
                            grannet = grannet)
            else:
                raise ValueError(" params['graph_params']['kernel_grannet_type'] is incorrect")
                
            # elif  params['graph_params']['kernel_grannet_type'] == 'averaged':
            #     # THIS OPTION AVERAGE ONE KERNEL WITH THE OTHER KERNELS 
            #     corr_kern = np.dstack([mkDataGraph(data[:,:,trial_counter], params, reduceDim = params['reduceDim'], 
            #                          reduceDimParams = {'alg':'PCA'}, graph_function = 'gaussian',
            #                 K_sym  = True, use_former = use_former, data_name = data_type, toNormRows = True, data = data, graph_params = params['graph_params'], grannet = grannet) for trial_counter in range(data.shape[2])])
        
         
            """
            RETURN TO REAL DIMENSION DATA!!!!!!!!!!!!!!!!!!!!!!!!!
            """
            if params['is_trials'] and params['is_trials_type'] == 'shared_flex':
                if (trials_counts == trials_counts[0]).all():
                     corr_kern = np.repeat(corr_kern, trials_counts[0], axis = 2)
                else:
                     corr_kern = np.repeat(corr_kern, trials_counts, axis = 2)
                data = data_full.copy()
            elif params['is_trials'] and params['is_trials_type'] == 'shared':
     
                data = data_full.copy()
             
            
                
                
                
            
            
            if to_save and not use_former:
                #path_name = data_type + os.sep + str(date.today()) + os.sep 
                if not os.path.exists(path_name):
                    os.makedirs(path_name)
                #path_name + os.sep +
                if len(path_name) > 0:
                    np.save(path_name + 'kernel_%s_xmin_%s_xmax_%s_ymin_%s_ymax_%s.npy'%(data_type, str(params[data_type]['xmin']),
                                                                              str(params[data_type]['xmax']),
                                                                              str(params[data_type]['ymin']),
                                                                              str(params[data_type]['ymax'])), corr_kern)
                    
                else:
                    np.save('kernel_%s_xmin_%s_xmax_%s_ymin_%s_ymax_%s.npy'%(data_type, str(params[data_type]['xmin']),
                                                                              str(params[data_type]['xmax']),
                                                                              str(params[data_type]['ymin']),
                                                                              str(params[data_type]['ymax'])), corr_kern)
                   
                
                print('kernel saved!')
                
                
                
        elif isinstance(corr_kern, str): # Check if path
          try:
              if len(path_name) > 0:
                  try:
                      corr_kern = np.load(path_naem + 'kernel_%s_xmin_%d_xmax_%d_ymin_%d_ymax_%d.npy'%corr_kern)
                  except:
                      corr_kern = np.load(path_name + 'kernel_%s_xmin_%d_xmax_%d_ymin_%d_ymax_%d.npy'%(data_type, params[data_type]['xmin'],params[data_type]['xmax'],params[data_type]['ymin'],params[data_type]['ymax']))
                      
              else:
                  try:
                      corr_kern = np.load('kernel_%s_xmin_%d_xmax_%d_ymin_%d_ymax_%d.npy'%corr_kern)
                  except:
                      corr_kern = np.load('kernel_%s_xmin_%d_xmax_%d_ymin_%d_ymax_%d.npy'%(data_type, params[data_type]['xmin'],params[data_type]['xmax'],params[data_type]['ymin'],params[data_type]['ymax']))
                      
          except:
              print(corr_kern)
              raise ValueError('Cannot locate kernel path! (your invalid path is %s)'%corr_kern)
        
        if params['is_trials'] and params['is_trials_type'] == 'shared':
            print('is here!!!!!!!!!!!!!!!!!!!!!!')
            if coefficients_similar_nets.shape[0] != len(np.unique(labels)):
                # in this case I except coefficients_similar_nets to be unique(k) X  unique(k) X nets
                raise ValueError('Dimensions mismatch. Data shape is '+ str(len(np.unique(labels))) + ' while coeffs shape is ' + str(coefficients_similar_nets.shape))

        elif data.shape[2] != coefficients_similar_nets.shape[0]:
            """
            if kernel is shared and nets are shared then __. 
            """
            print(params['is_trials'] )
            print(params['is_trials_type'] == 'shared')
        
            raise ValueError('Dimensions mismatch. Data shape is '+ str(data.shape) + ' while coeffs shape is ' + str(coefficients_similar_nets.shape))

    else: 
        """
        graft case (not grannet)
        """

        #print(corr_kern)
        if checkEmptyList(corr_kern):
            #print('fddddddddddddddddddddddddddddddddddddddddddddddddddddd')
            print('IT IS EMPTY LIST!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
      
            corr_kern  = mkDataGraph(data, params, reduceDim = params['reduceDim'], 
                                 reduceDimParams = {'alg':'PCA'}, graph_function = 'gaussian',
                        K_sym  = True, use_former = False, data_name = data_type, toNormRows = True)
            if to_save: 
                if not os.path.exists(path_name):
                    os.makedirs(path_name)
            
                np.save(path_name + 'kernel_%s_xmin_%s_xmax_%s_ymin_%s_ymax_%s.npy'%(data_type, str(params[data_type]['xmin']),
                                                                              str(params[data_type]['xmax']),
                                                                              str(params[data_type]['ymin']),
                                                                              str(params[data_type]['ymax'])), corr_kern)
                
                print('kernel saved!')
            
            
        elif isinstance(corr_kern, str): # Check if path
          try:
              print('kernel_%s_xmin_%d_xmax_%d_ymin_%d_ymax_%d.npy'%(data_type,params[data_type]['xmin'],params[data_type]['xmax'],params[data_type]['ymin'],params[data_type]['ymax']))
              if len(path_name) > 0:
                  corr_kern = np.load(path_name + 'kernel_%s_xmin_%d_xmax_%d_ymin_%d_ymax_%d.npy'%(data_type,params[data_type]['xmin'],params[data_type]['xmax'],params[data_type]['ymin'],params[data_type]['ymax']))

              else:
                  corr_kern = np.load('kernel_%s_xmin_%d_xmax_%d_ymin_%d_ymax_%d.npy'%(data_type,params[data_type]['xmin'],params[data_type]['xmax'],params[data_type]['ymin'],params[data_type]['ymax']))
              #print('TYPE CORRKERN11111111111111111111')
              #print(type(corr_kern))
          except:
              raise ValueError('Cannot locate kernel path! (your invalid path is %s)'%corr_kern)
        else:
            raise ValueError('Kernel should be an empty list or str. Currently it is a ' + str(type(corr_kern)))
    #print(corr_kern)
    #print(type(corr_kern))
    #print('IT IS EMPTY LIST!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    #raise ValueError('metumtam!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            
    """
    save params
    """
    if to_save:
        #path_name = data_type + os.sep + str(date.today()) + os.sep 
        if not os.path.exists(path_name):
            os.makedirs(path_name)
        np.save(path_name + os.sep +'all_params.npy', locals())
    
              
    """
    run graft
    """
    
    if to_save or save_mid_results:
        if not os.path.exists(path_name):
            os.makedirs(path_name)
        np.save(path_name + 'params.npy',params )
    if params['usePatch']:
        raise ValueError('Use Patch is not available yet!')
        
    elif grannet:
        # full_phi = {}
        # full_A = {}
        # additional_return  ={}
        # error_list = []
        full_phi, full_A, additional_return, error_list = GraFT_with_GraNNet(data, [],corr_kern, params,  coefficients_similar_nets = coefficients_similar_nets
                            , grannet = True,  seed = 0,  to_store_lambdas = params['to_store_lambdas'],
                            save_mid_results = save_mid_results, dataname = data_type, save_name = save_name, 
                            path_name = path_name, labels = labels)
        
    else:
        phi, A, additional_return, error_list = GraFT_with_GraNNet(data, [],corr_kern, params,  
                                                                 coefficients_similar_nets =[]
                           , grannet = False,  seed = 0,  to_store_lambdas = params['to_store_lambdas'],
                           save_mid_results=save_mid_results, dataname = data_type, save_name = save_name, 
                           path_name = path_name, labels = labels)
        #[phi, A, additional_return] = GraFT(data, [], corr_kern, params)                    # Learn the dictionary (no patching - will be much more memory intensive and slower)
    
    if to_save:
        #save_name = input('save_name')
        #path_name = 'date_'+ str(date.today()) + '_xmin_%d_xmax_%d_ymin_%d_ymax_%d.npy'%(params['xmin'],params['xmax'],params['ymin'],params['ymax'])
        
        #path_name = data_type + os.sep + str(date.today()) + os.sep 
        if not os.path.exists(path_name):
            os.makedirs(path_name)
        if grannet:
            np.save(path_name + os.sep + save_name + '.npy', {'phi':full_phi, 'A':full_A, 'data':data, 'params':params, 'divide_med':params['divide_med'], 
                                                           'usePatch':params['usePatch'], 'shape':data.shape, 'additional': additional_return, 
                                                           'save_name':save_name, 'error_list': error_list})
        else:
            try:
                np.save(path_name + os.sep + save_name + '.npy', {'phi':phi, 'A':A, 'data':data, 'params':params, 'divide_med':params['divide_med'], 
                                                               'usePatch':params['usePatch'], 'shape':data.shape, 'additional': additional_return, 
                                                               'save_name':save_name, 'error_list': error_list})
            
            except:
                np.save(path_name + os.sep + save_name_short + '.npy', {'phi':phi, 'A':A, 'data':data, 'params':params, 'divide_med':params['divide_med'], 
                                                               'usePatch':params['usePatch'], 'shape':data.shape, 'additional': additional_return, 
                                                               'save_name':save_name,'error_list':error_list})
                    

    if to_return:
        print(grannet)
        if grannet:
            return full_A, full_phi, additional_return
        return A, phi, additional_return

def init_mat(shape, dist_init, multi = 1, params_dist_int = {}):
    """
    This function initializes a matrix with a specified distribution.
    
    Parameters:
    shape (tuple): The shape of the matrix to be initialized.
    dist_init (str): The type of distribution to use for initialization.
                    Options: 'zeros', 'rand', 'uniform', 'normal'
    multi (int, optional): A multiplier for the matrix values. Default value is 1.
    params_dist_int (dict, optional): Additional parameters for the distribution. 
                                      Default value is an empty dictionary.
                                      Only relevant for 'normal' distribution.
                                      The default values for loc and scale are 0 and 1.0 respectively.
                                      
    Returns:
    ndarray: The initialized matrix.
    
    Example:
    >>> init_mat((3,3), 'zeros', 2)
    array([[0., 0., 0.],
           [0., 0., 0.],
           [0., 0., 0.]])
    >>> init_mat((3,3), 'normal', 2, {'loc':0, 'scale':1.0})
    array([[ 1.70956002, -0.13946417,  1.49056311],
           [ 0.60136805, -1.00341437,  2.572688  ],
           [-0.25894951,  0.47257538,  1.52980708]])
    """
    
    params_dist_int = {**params_dist_int, **{'loc':0, 'scale': 1.0}}
    if dist_init == 'zeros':
        A = np.zeros(shape)# np.zeros((n_neurons, p))
    elif  dist_init == 'rand':
        A = np.random.rand(*shape) * multi
    elif  dist_init == 'randn':
        A = np.random.randn(*shape) * multi
    elif dist_init == 'uniform':
        A = np.random.uniform(0, 1,size =shape )*multi    
    elif dist_init == 'normal':
        A = np.random.normal(params_dist_int['loc'], params_dist_int['scale'], size = shape)*multi  
    
    return A


def norm_to_plot(mat_2d, epsilon = 0.01):
    return np.hstack([norm_vec_min_max(mat_2d[:,t]).reshape((-1,1)) for t in range(mat_2d.shape[1])]) 
# herehere
# def norm_to_plot(mat_2d, epsilon = 0.01):
#     return np.hstack([(mat_2d[:,t]/(epsilon + np.max(mat_2d[:,t]))).reshape((-1,1)) for t in range(mat_2d.shape[1])])  
 
def norm_vec_min_max(vec)  :
    return (vec - np.min(vec))/(vec.max() - np.min(vec))
    
          
def GraFT_with_GraNNet(data, phi, kernel, params,  coefficients_similar_nets = [], grannet = True,  seed = 0,
                       to_store_lambdas = params_default['to_store_lambdas'], 
                       save_mid_results = True, path_save = '', T_save = 10, dataname = 'unknown', 
                       save_name = 'def', path_name = '', labels = []):
    """
    Function to learn a dictioanry for spatially ordered/ graph-based data using a
    re-weighted l1 spatial / graph filtering model.
    
    Parameters
    ----------
    data : np.array
        neurons X time OR (for grannet:)  neurons X time X trials
    phi : TYPE
        (time, p)
    kernel : neurons X neurons
        DESCRIPTION.
    params : TYPE
        DESCRIPTION.
    coefficients_similar_nets = [] - for further applications (GraNNet)
    coefficients_similar_nets = trials X trials X #net matrix describing the PROXIMITY between nets (i.e. higher values = higer reg.)- for further applications (GraNNet)
    T_save: frequency of saving
    
    Returns
    -------
    None.
    
    """
    #n_neurons = data.shape[0]
    #n_times = data.shape[1]
    np.random.seed(seed)
    if grannet and len(data.shape) != 3: 
        raise ValueError('Data Should be a 3-dim array in case of GraNNet!')
    additional_return  = {'MSE':[]}
    if len(data.shape) == 3 and not grannet: 
        data = MovToD2(data)
    params = {**{'max_learn': 1e3, 'learn_eps': 0.01,'step_decay':0.995}, **params}
    #params = createDefaultParams(params)
    n_rows = data.shape[0] # number of neurons, N
    n_cols = params['p']# data.shape[1], p
    #p = params['p']
    n_times = data.shape[1]
    extras = {'dictEvo':[], 'presMap':[], 'wghMap':[]} # Store some outputs
    
    """
    Initialize dictionary
    """

    
    if params['to_sqrt']:
        multi = np.sqrt(np.mean(data))
    else:
        multi = 1
       
    step_GD = params['step_s']    
    n_iter = 0
    error_dict = np.inf
    cur_error = np.inf #
    
    """
    Define Variables Specific for GraNNet
    """
    if grannet: 
        if params['is_trials'] and params['is_trials_type'] == 'shared':
            if len(np.unique(labels)) != kernel.shape[2]:
                raise ValueError("len(np.unique(labels)) != corr_kern.shape[2], %s"%str(len(np.unique(labels)) ,kernel.shape[2]))
            n_states = kernel.shape[2]
            print('n states:')
            print(n_states)
        else:
            n_states = data.shape[2]
        n_states_phi = data.shape[2]
        
        """
        initialize A
        """
        if params['dist_init'] == 'zeros':
            full_A = np.zeros((n_rows, n_cols, n_states))
            if to_store_lambdas:  full_lambdas = np.zeros((n_rows, n_cols, n_states))
        elif params['dist_init'] == 'rand':
            full_A = np.random.rand(n_rows, n_cols, n_states)*multi
            if to_store_lambdas: full_lambdas = np.random.rand(n_rows, n_cols, n_states)*multi
        elif params['dist_init'] == 'randn':
            full_A = np.random.randn(n_rows, n_cols, n_states)*multi
            if to_store_lambdas: full_lambdas = np.random.randn(n_rows, n_cols, n_states)*multi
        elif params['dist_init'] == 'uniform':
            full_A = np.random.uniform(0, 1,size = (n_rows, n_cols, n_states))*multi
            if to_store_lambdas: full_lambdas = np.random.uniform(0, 1,size = (n_rows, n_cols, n_states))*multi        
        else:
            raise ValueError('Unknown dist init')
            
        """
        initialize phi    
        """
        full_phi = np.dstack([dictInitialize(phi, (n_times, n_cols), params = params) for _ in range(n_states_phi)])
        full_phi = full_phi * multi 
        
    else:
        if params['dist_init'] == 'zeros':
            A = np.zeros((n_rows, n_cols))
            if to_store_lambdas:  lambdas = np.zeros((n_rows, n_cols))
        elif params['dist_init'] == 'rand':
            A = np.random.rand(n_rows, n_cols)*multi
            if to_store_lambdas: lambdas = np.random.rand(n_rows, n_cols)*multi
        elif params['dist_init'] == 'uniform':
            A = np.random.uniform(0, 1,size = (n_rows, n_cols))*multi
            if to_store_lambdas: lambdas = np.random.uniform(0, 1,size = (n_rows, n_cols))*multi  
        elif params['dist_init'] == 'randn':
            A = np.random.randn(n_rows, n_cols) * multi
        else:
            raise ValueError('Unknown dist init')

        phi = dictInitialize(phi, (n_times, n_cols), params = params)
        phi = phi * multi 
        
        
    """
    starting the EM algorithm
    """
    error_list = []
    
    """
    to load real to compare?
    """
    if params['compare_to_synth'] and 'synth_grannet' in str(params['data_name']):
        d_full = full_synth_grannet = np.load(r'grannet_synth_results_march_2023.npy', allow_pickle=True).item()
        A_real = d_full['A']
        phi_real = d_full['phi']
        Y_real = d_full['Y']
        
    if params['num_nets_sample_trials'] != 0  :
        full_A_original = full_A.copy()
        full_phi_original = full_phi.copy()
        raise ValueError('not fully implemented yet! need to do the actualy choosing later - future work futuresteps')
        
    while n_iter < params['max_learn'] and (error_dict > params['dict_max_error'] or cur_error > params['mean_square_error']):
        print('iter')
        n_iter += 1
        print(n_iter)
        
        
        """
        CHOOSE NETS and stack trials
        """
        if params['num_nets_sample_trials'] != 0:
            if not params['is_trials'] :
                raise ValueError('only for trials case! please set is_trials to True')
            
            inds_per_cond = [np.random.choice(np.min([params['num_nets_sample_trials'], trial_count ]), replace = False  ) 
                             for trial_count in params['trials_counts'] ]
            raise ValueError('not implemented yet!')
            
        if params['is_trials'] and params['is_trials_type'] == 'shared': # before the update of A
            full_A_original = full_A.copy()
            full_phi_original = full_phi.copy()           
            data_original = data.copy()
            # (to recover with split_stacked_data(full_phi.transpose((0,1,2)), T = full_phi_original.shape[0], k = full_phi_original.shape[2]).transpose((0,1,2)) )
            #full_phi = MovToD2_grannet(full_phi.transpose((1,0,2)))
            full_phi = np.dstack([MovToD2_grannet(full_phi[:,:,labels == k].transpose((1,0,2))).T
                                  for  k in params['lab_unique']])
            print(full_phi.shape)
        
            data = np.dstack([MovToD2_grannet(data[:,:,labels == k]) for  k in params['lab_unique']])
            
            
        
            
        """
        compute the presence coefficients from the dictionary:
            CLACULATE A
        """
      
        if grannet: 
            if n_iter > params['grannet_params']['late_start']:
                prox_nets = coefficients_similar_nets.copy()
            else:
                prox_nets = np.zeros(coefficients_similar_nets.shape)
                
            for trial_num in range(n_states):
                cur_error_before = np.mean((full_A[:,:,trial_num] @ full_phi[:,:,trial_num].T - data[:,:,trial_num])**2) 

                A, lambdas = dictionaryRWL1SF(data[:,:,trial_num], full_phi[:,:,trial_num], kernel[:,:,trial_num], 
                                              params = params, 
                                              A = full_A[:,:,trial_num],  grannet =  grannet,  
                                              coefficients_similar_nets = prox_nets[trial_num],
                                              trial_num = trial_num, full_A = full_A) 
                cur_error = np.mean((A @ full_phi[:,:,trial_num].T - data[:,:,trial_num])**2) 

                # Infer coefficients given the data and dictionary
                if (params['A_only_dec'] and cur_error < cur_error_before) or not params['A_only_dec'] :
                    full_A[:,:,trial_num] = A
                    if to_store_lambdas: full_lambdas[:,:,trial_num] = lambdas
                else:
                    print('did not update A since error increased')
            if A.mean() < 1e-29:
                A = A + init_mat(shape = A.shape, dist_init = 'normal', multi = np.median(data))
        else:      
            A, lambdas = dictionaryRWL1SF(data, phi, kernel, params = params, A=A) # Infer coefficients given the data and dictionary
            #print(A.mean(0))
            #print(lambdas)
            if A.mean() < 1e-29:
                A = A + init_mat(shape = A.shape, dist_init = 'normal', multi = np.median(data))
                print('A mean after noise normal')
                print(str(A.mean()))
            #if n_iter > 100:
            #    raise ValueError('gfgf')
            #raise ValueError('dfdsdf')
        """
        take hard thres on A
        """
        if params['hard_thres'] and cur_error < params['hard_thres_params']['thres_error_hard_thres'] and np.mod(n_iter, params['hard_thres_params']['T_hard']) == 0:
            if grannet: 
                A = np.dstack([hard_thres_on_A(full_A[:,:,k], params['hard_thres_params']['non_zeros']) for k in range(full_A.shape[2])])
            else:
                
                A = hard_thres_on_A(A, params['hard_thres_params']['non_zeros']) 
        

            

            
                
        if grannet: 
            
            if params['is_trials'] and params['is_trials_type'] == 'shared': # for updating phi

                
                if (params['trials_counts'] == params['trials_counts'][0]).all():
                    full_phi = split_stacked_data(full_phi.transpose((1,0,2)), T = full_phi_original.shape[0],
                                                  k = params['trials_counts'][0] ).transpose((1,0,2)) 
                    #data = np.dstack([MovToD2_grannet(data[:,:,labels == k]) for  k in params['lab_unique']])
                else:
                    raise ValueError('future direction - unequal trial counts')
                data = data_original.copy()
                
                full_A_original = full_A.copy()
                full_A = np.repeat(full_A, params['trials_counts'], axis = 2)
                

                
                
            """
            update phi
            """
            dict_old = full_phi[:,:,trial_num] # Save the old dictionary for metric calculations
            """
            find phi by inverse
            """
            
            if params['add_inverse'] and np.mod(n_iter, params['inverse_params']['T_inverse']) == 0:
                for trial_num in range(n_states_phi):
                    phi = data[:,:,trial_num].T @ np.linalg.pinv(full_A[:,:,trial_num]).T 
                    full_phi[:,:,trial_num] = phi
            else:    
                if params['various_steps']:
                    steps_opts = step_GD*params['steps_range']
                    step_selected_list = []
                    for trial_num in range(n_states_phi):
                        cur_error_list = []
                        phi_list = []

                        cur_error_before = np.mean((full_A[:,:,trial_num] @ full_phi[:,:,trial_num].T - data[:,:,trial_num])**2) 
                        for step_opt in steps_opts:
                             
                            phi = dictionary_update(full_phi[:,:,trial_num], full_A[:,:,trial_num], data[:,:,trial_num], 
                                                    step_opt, GD_type = params['GD_type'], params = params) # Take a gradient step with respect to the dictionary  
                            cur_error = np.mean((full_A[:,:,trial_num] @ phi.T - data[:,:,trial_num])**2)  
                            cur_error_list.append(cur_error)
                            phi_list.append(phi)
                        if params['step_loc'] == 0:
                            argmin = np.argmin(cur_error_list)
                        else:
                            sortedarg_cur_error_list = np.argsort(cur_error_list)
                            argmin = sortedarg_cur_error_list[params['step_loc']]
                        print('argmin is!!!!!!!!!!!!!!!!!!!!!!')
                        print(argmin)
                        print('!!!!!!!!!!!!!!!!!!!!!!!')
                        phi_selected = phi_list[argmin]
                        if cur_error_list[argmin] < cur_error_before + cur_error_before*params['min_add_phi_error'] or not params['phi_only_dec']:
                            full_phi[:,:,trial_num] = phi_selected
                        else:
                            print(params['phi_only_dec'])
                            #input('ok?')
                            print('did not change phi since error increased')
                        step_selected_list.append(steps_opts[argmin]  )
            
                    step_GD   = np.median(step_selected_list )   
                    print('step GD:')
                    print(step_GD)
                    print('           ')
                    step_GD   = step_GD*params['step_decay']     
                else:
                    for trial_num in range(n_states_phi):
                        phi = dictionary_update(full_phi[:,:,trial_num], full_A[:,:,trial_num], data[:,:,trial_num], 
                                                step_GD, GD_type = params['GD_type'], params = params) # Take a gradient step with respect to the dictionary     
                        full_phi[:,:,trial_num] = phi
                    step_GD   = step_GD*params['step_decay']                                                     # Update the step size 
            if step_GD < params['min_step_size']:
                step_GD = params['min_step_size']
            if step_GD > params['max_step_size']:
                step_GD = params['max_step_size']
                
            error_dict    = norm((full_phi[:,:,trial_num] - dict_old).flatten())/norm(dict_old.flatten())
            cur_error  = np.mean([np.mean((full_A[:,:,trial_num] @ full_phi[:,:,trial_num].T - data[:,:,trial_num])**2)  for trial_num in range(n_states)])                                                # store error                   # Calculate the difference in dictionary coefficients
            additional_return['MSE'].append(cur_error)            
            
            if params['add_noise_stuck'] and len(additional_return['MSE']) > 2*params['noise_stuck_params']['in_a_row']:
                last_errors = additional_return['MSE'][-params['noise_stuck_params']['in_a_row']:]
                full_A, full_phi, step_GD = add_noise_if_stuck(last_errors, full_A, full_phi,  step_GD, params_stuck = params['noise_stuck_params'])
                params['noise_stuck_params']['std_noise_A'] *= 0.99
                params['noise_stuck_params']['std_noise_phi'] *= 0.99
                
            
            if params['is_trials'] and params['is_trials_type'] == 'shared': # after update phi
                full_A = full_A_original.copy()
                
            # dict_old = full_phi[:,:,trial_num] # Save the old dictionary for metric calculations
            # for trial_num in range(n_states):
            #     if params['add_inverse'] and np.mod(n_iter, params['inverse_params']['T_inverse']) == 0:
            #         phi = data[:,:,trial_num].T @ np.linalg.pinv(full_A[:,:,trial_num]).T 
            #         full_phi[:,:,trial_num] = phi
            #     else:    
            #         if params['various_steps']:
            #             steps_opts = step_GD*params['steps_range']
            #             for step_opt in steps_opts:
            #                 phi = dictionary_update(full_phi[:,:,trial_num], full_A[:,:,trial_num], data[:,:,trial_num], 
            #                                         step_opt, GD_type = params['GD_type'], params = params) # Take a gradient step with respect to the dictionary  
            #                 e = herehere
            #         else:
            #             phi = dictionary_update(full_phi[:,:,trial_num], full_A[:,:,trial_num], data[:,:,trial_num], 
            #                                     step_GD, GD_type = params['GD_type'], params = params) # Take a gradient step with respect to the dictionary     
            #             step_GD   = step_GD*params['step_decay']                                                     # Update the step size    
                        
                     
            #         full_phi[:,:,trial_num] = phi
                    
            #     error_dict    = norm((full_phi[:,:,trial_num] - dict_old).flatten())/norm(dict_old.flatten())
            #     cur_error  = np.mean((full_A[:,:,trial_num] @ full_phi[:,:,trial_num].T - data[:,:,trial_num])**2)                                                  # store error                   # Calculate the difference in dictionary coefficients
            #     additional_return['MSE'].append(cur_error)
            
            """
            save results
            """
            
            """
            Second step is to update the dictionary:
            """
            if ((params['save_plots']  or save_mid_results) and np.mod(n_iter, T_save) == 0) or (params['compare_to_synth'] and np.mod(n_iter, params['T_save_fig']) == 0):
                name_save = 'iter_%d_error100_%s'%(n_iter, str(cur_error*100).replace('.','dot'))
                if len(path_name) ==0:
                    path_name = dataname + os.sep + str(date.today()) + os.sep 
                path_save =   path_name + os.sep  +  'mid_results' + os.sep
                if (not os.path.exists(path_save)) and save_mid_results and np.mod(n_iter, T_save) == 0:
                    os.makedirs(path_save)
                    print('creates path to save...')     
                
                
            if (save_mid_results and np.mod(n_iter, T_save) == 0) or (params['compare_to_synth'] and np.mod(n_iter, params['T_save_fig']) == 0):
                print('save results ...')   
                try:
                    Y_reco 
                except:
                    if full_A.shape[2] == full_phi.shape[2]:
                        Y_reco = np.dstack([ full_A[:,:,k] @ full_phi[:,:,k].T   for k in range(full_phi.shape[2])])
                    else:
                        full_A_rep = np.repeat(full_A, params['trials_counts'], axis = 2)
                        Y_reco = np.dstack([ full_A_rep[:,:,k] @ full_phi[:,:,k].T   for k in range(full_phi.shape[2])])
                np.save( path_save +  name_save  + '.npy', 
                        {'params':params,'additional_return': additional_return, 'full_phi': full_phi,
                         'full_A':full_A, 'data':data, 'save_name':save_name, 'cur_error':cur_error, 'Y_reco': Y_reco})
            if params['save_plots'] and np.mod(n_iter, T_save) == 0 and np.mod(n_iter, params['T_save_fig']) == 0:
  
                
                fig, axs = plt.subplots(2, full_A.shape[2], figsize = (20,10)) 
                [sns.heatmap(full_A[:,:,k], ax = axs[0,k], robust = True) for k in range(full_A.shape[2])] 
                [sns.heatmap(full_phi[:,:,k], ax = axs[1,k], robust = True) for k in range(full_A.shape[2])]
                plt.savefig(path_save +  name_save + '.png')
                    
                plt.close()
           
            if params['compare_to_synth'] and np.mod(n_iter, params['T_save_fig']) == 0:
                print('before ordering')                
                full_phi_ordered, full_A_ordered = snythetic_evaluation(full_A, full_phi, A_real, phi_real)
                print('after ordering')
                
                print('before shared figure')   
                fig, axs = plt.subplots(7,len(A_real), figsize = (20,10)) 
                [sns.heatmap(norm_to_plot(A_real[k+1]), ax = axs[0,k], robust = True) for k in range(len(d_full['A']))] 
                [sns.heatmap(norm_to_plot( full_A_ordered[:,:,k]), ax = axs[1,k], robust = True) for k in range(full_A_ordered.shape[2])] 
                [sns.heatmap(full_phi_ordered[:,:,k], ax = axs[3,k], robust = True) for k in range(full_A_ordered.shape[2])]
                [sns.heatmap(phi_real[k+1], ax = axs[2,k], robust = True) for k in range(len(d_full['A']))]
                
                # Y reco
                Y_reco = np.dstack([ full_A_ordered[:,:,k] @ full_phi_ordered[:,:,k]   for k in range(full_phi_ordered.shape[2])])
                [sns.heatmap(Y_reco[:,:,k], ax = axs[5,k], robust = True) for k in range(full_A_ordered.shape[2])]
                [sns.heatmap(Y_real[k+1], ax = axs[4,k], robust = True) for k in range(len(d_full['A']))]
                
                [sns.heatmap(Y_real[k+1] - Y_reco[:,:,k], ax = axs[6,k], robust = True) for k in range(len(d_full['A']))]
                
                ylabels = ['A real', 'A reco', 'phi real', 'phi reco','Y real', 'Y reco' , 'Y diff' ]
                [ax_s.set_ylabel(ylabels[count]) for count, ax_s in enumerate(axs[:,0])]
                [axs[0,counter].set_title('condition %d'%k) for counter, k in enumerate(list(d_full['A'].keys()))]
                fig.tight_layout()
                

                # ylabels = ['A real', 'A reco', 'phi real', 'phi reco' ]
                # [ax_s.set_ylabel(ylabels[count]) for count, ax_s in enumerate(axs[:,0])]
                plt.savefig(path_save +  name_save + '_compare_to_real.png')
                    
                plt.close()
                print('after shared figure')   
                
                """
                plot side by size
                """
                   
                fig, axs = plt.subplots(2,len(A_real), figsize = (20,10))
                for counter, (key, val) in enumerate(A_real.items()):
                    ax = axs[0,counter]
                    A1 = A_real[key]
                    A2 = full_A_ordered[:,:,counter]
                    plot_nets_side_by_size(A1,A2, 1, ax = ax)
                    ax = axs[1,counter]
                    phi1 = phi_real[key]
                    phi2 = full_phi_ordered[:,:,counter]
                    plot_nets_side_by_size(phi1,phi2, 0, ax = ax)
                    
                [ax_s.set_xticklabels(['r','g']*A1.shape[1]) for ax_s in axs[0]]
                [ax_s.set_yticklabels(['r','g']*phi1.shape[0]) for ax_s in axs[0]]
                [ax_s.set_yticklabels(['r','g']*phi1.shape[0]) for ax_s in axs[1]]
                [axs[j,0].set_ylabel(lab) for j, lab in enumerate(['A','phi'])]

                [axs[0,j].set_title(lab) for j, lab in enumerate(['cond %d'%k for k in A_real.keys()])]

                plt.savefig(path_save +  name_save + '_one_one_compare.png')
                    
                plt.close()

                    
        else:
            dict_old = phi # Save the old dictionary for metric calculations
            phi = dictionary_update(phi, A, data, step_GD, GD_type = params['GD_type'], params = params) # Take a gradient step with respect to the dictionary      

            cur_error  = np.mean((A @ phi.T - data)**2)                                                  # store error
            
            additional_return['MSE'].append(cur_error)
            #print('mean A here is ')
            # print(A.mean())
            # print('mean phi here is ')
            # print(phi.mean())
            
            if save_mid_results and np.mod(n_iter, T_save) == 0:
                try:
                    name_save
                except:
                    name_save = 'iter_%d_error100_%s'%(n_iter, str(cur_error*100).replace('.','dot'))
                    if len(path_name) ==0:
                        path_name = dataname + os.sep + str(date.today()) + os.sep 
                    path_save =   path_name + os.sep  +  'mid_results' + os.sep
                    if (not os.path.exists(path_save)) and save_mid_results and np.mod(n_iter, T_save) == 0:
                        os.makedirs(path_save)
                        print('creates path to save...')   
                
                # print('mean A')
                # print(A.mean())
                print('save results ...')
                print('loc')
                print(path_save + name_save  + '.npy')
                np.save( path_save + name_save  + '.npy', 
                        {'params':params, 'additional_return': additional_return, 'phi': phi, 'A':A, 'data':data,
                         'save_name':save_name, 'cur_error':cur_error, 'Y_reco': Y_reco})
            print('...')
        params['l3'] = params['lamContStp']*params['l3'];                                            # Continuation parameter decay
        print('Current Error is: {:.2f}'.format(cur_error))
        error_list.append(cur_error)




    
    """
    post-processing
    """
    # Re-compute the presence coefficients from the dictionary:
    if params['normalizeSpatial']:
        if grannet:
            for trial_num in range(n_states):
                
                A, lambdas = dictionaryRWL1SF(data[:,:,trial_num], phi[:,:,trial_num], kernel, 
                                              params = params, 
                                              A = full_A[:,:,trial_num],  grannet =  grannet,  
                                              coefficients_similar_nets = coefficients_similar_nets,
                                              trial_num = trial_num, full_A = full_A) 
                # Infer coefficients given the data and dictionary
                full_A[:,:,trial_num] = A
                if to_store_lambdas: full_lambdas[:,:,trial_num] = lambdas            
        else:
            A, lambdas = dictionaryRWL1SF(data, phi, kernel, params, A)   # Infer coefficients given the data and dictionary
            
    if params['reorder_nets_by_importance']:
        if grannet:
            """
            No re-organization, since the networks need to remain in the same space
            """
            pass
                
        else:
            Dnorms   = np.sqrt(np.sum(phi**2,0))               # Get norms of each dictionary element
            Smax     = np.max(A,0)                                                    # Get maximum value of each spatial map
            actMeas  = Dnorms*Smax                                             # Total activity metric is the is the product of the above        
            IX   = np.argsort(actMeas)[::-1]       # Get the indices of the activity metrics in descending order
            phi = phi[:,IX]                                                 # Reorder the dictionary
            A   = A[:,IX]                                                        # Reorder the spatial maps
            
    if grannet:
        return full_phi, full_A, additional_return, error_list
    return phi, A, additional_return, error_list

    
def GraFT(data, phi, kernel, params,  coefficients_similar_nets = [], grannet = False):
    """
    Function to learn a dictioanry for spatially ordered/ graph-based data using a
    re-weighted l1 spatial / graph filtering model.
    
    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    phi : TYPE
        DESCRIPTION.
    kernel : TYPE
        DESCRIPTION.
    params : TYPE
        DESCRIPTION.
    coefficients_similar_nets = trials X trials X #net matrix describing the PROXIMITY between nets (i.e. higher values = higer reg.)- for further applications (GraNNet)
    
    Returns
    -------
    None.
    
    """

    additional_return  = {'MSE':[]}
    if len(data.shape) == 3: data = MovToD2(data)
    params = {**{'max_learn': 1e3, 'learn_eps': 0.01,'step_decay':0.995}, **params}
    #params = createDefaultParams(params)
    n_rows = data.shape[0] # number of neurons
    n_cols =params['p']# data.shape[1]
    n_times = data.shape[1]
    extras = {'dictEvo':[], 'presMap':[], 'wghMap':[]} # Store some outputs
    
    #%% Initialize dictionary

    phi = dictInitialize(phi, (n_times, n_cols), params = params)
    if params['multi'] == 'med':
        multi = np.median(data)
    elif params['multi'] == 'sqrt':
        multi = np.sqrt(np.mean(data))
    elif params['multi'] == 'none': 
        multi = 1
    phi = phi * multi    
    step_GD = params['step_s']
    
    lambdas = []  # weights
    A = []
    
    n_iter = 0
    error_dict = np.inf
    cur_error = np.inf #
    if grannet: full_A = np.zeros()
    while n_iter < params['max_learn'] and (error_dict > params['dict_max_error'] or cur_error > params['mean_square_error']):
        
        n_iter += 1
        """
        compute the presence coefficients from the dictionary:
        """
        
        A, lambdas = dictionaryRWL1SF(data, phi, kernel, params = params, A=A) # Infer coefficients given the data and dictionary


        """
        Second step is to update the dictionary:
        """
        dict_old = phi # Save the old dictionary for metric calculations

        phi = dictionary_update(phi, A, data, step_GD, GD_type = params['GD_type'], params = params) # Take a gradient step with respect to the dictionary

        #raise ValueError('fdgfdgfdgfdgdfgsdfg')
        step_GD   = step_GD*params['step_decay']                                   # Update the step size

        error_dict    = norm((phi - dict_old).flatten())/norm(dict_old.flatten());        
        # Calculate the difference in dictionary coefficients
    
        params['l3'] = params['lamContStp']*params['l3'];                     # Continuation parameter decay
        cur_error  = np.mean((A @ phi.T - data)**2)
        additional_return['MSE'].append(cur_error)
        print('Current Error is: {:.2f}'.format(cur_error))
    ##  post-processing
    # Re-compute the presence coefficients from the dictionary:
    if params['normalizeSpatial']:
        A, lambdas = dictionaryRWL1SF(data, phi, kernel, params,A)   # Infer coefficients given the data and dictionary

    Dnorms   = np.sqrt(np.sum(phi**2,0))               # Get norms of each dictionary element
    Smax     = np.max(A,0)                                                    # Get maximum value of each spatial map
    actMeas  = Dnorms*Smax                                             # Total activity metric is the is the product of the above
    IX   = np.argsort(actMeas)[::-1]       # Get the indices of the activity metrics in descending order
    phi = phi[:,IX]                                                 # Reorder the dictionary
    A   = A[:,IX]                                                        # Reorder the spatial maps
    

    return phi, A, additional_return

def build_svd_effect(mat,  to_plot= True, max_comp_plot = 20 ,ax_heat = [], fig_heat = [], ax_error = [], fig_error = []):
    u, s,vt = np.linalg.svd(mat.astype(np.float32), full_matrices=True);
    num_practice = np.min([len(s), max_comp_plot])
    if checkEmptyList(ax_heat) and to_plot: fig_heat, ax_heat = plt.subplots(1, num_practice)
    if checkEmptyList(ax_error) and to_plot: fig_error, ax_error = plt.subplots()
    errors_list = []
    reco  = np.zeros(mat.shape)
    if to_plot:
        for j in range(num_practice):
            reco_temp = u[:,j].reshape((-1,1)) @ vt[j,:].reshape((1,-1)) * s[j]
            
            reco  += reco_temp
            sns.heatmap(reco, ax = ax_heat[j] )
            errors_list.append(np.mean(reco - mat)**2)
        ax_error.bar(np.arange(len(errors_list)) , errors_list)
        
        fig,ax = plt.subplots(); sns.heatmap(np.diag(s), ax = ax, robust = True); plt.title('s')

    return s
    
    
    
    
    
    
    
    
def norm(mat):
    """
    Parameters
    ----------
    mat : np.ndarray
        l2 norm of mat.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if len(mat.flatten()) == np.max(mat.shape):
        return np.sqrt(np.sum(mat**2))
    else:
        _, s, _ = np.linalg.svd(mat, full_matrices=True)
        return np.max(s)
    
def mkCorrKern(params = {}):
    """
    Parameters
    ----------
    params : TYPE, optional
        DESCRIPTION. The default is {}.

    Returns
    -------
    corr_kern : TYPE
        DESCRIPTION.

    """
    # Make a kernel
    raise ValueError('not clear, depracated')
    params = {**{'w_space':3,'w_scale':4,'w_scale2':0.5, 'w_power':2,'w_time':0}, **params}
    dim1  = np.linspace(-params['w_scale'], params['w_scale'], 1+2*params['w_space']) # space dimension
    dim2  = np.linspace(-params['w_scale2'], params['w_scale2'], 1+2*params['time']) # time dimension
    corr_kern  = gaussian_vals(dim1, std = params['w_space'], mean = 0 , norm = True, 
                               dimensions = 2, mat2 = dim2, power = 2)
    return corr_kern
    
def checkCorrKern(data, corr_kern, param_kernel = 'embedding', recreate = False, know_empty = False):
    if len(corr_kern) == 0: #raise ValueError('Kernel cannot ')
        if not know_empty: warnings.warn('Empty Kernel - creating one')
        if param_kernel == 'embedding' and recreate:
            corr_kern  = mkDataGraph(data, corr_kern) 
        elif  param_kernel == 'convolution'  and recreate:
            corr_kern  = mkCorrKern(corr_kern) 
        else:
            raise ValueError('Invalid param_kernel. Should be "embedding" or "convolution"')
            
    return corr_kern

    

def checkEmptyList(obj):
    return isinstance(obj, list) and len(obj) == 0
    
def dictionaryRWL1SF(data, phi, corr_kern, A = [], params = {}, grannet = False, coefficients_similar_nets = [],
                     trial_num = -1, full_A = [], initial_round_free = True,):
    """
    Updating A by gradient descent
    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    phi : TYPE
        DESCRIPTION.
    corr_kern : TYPE
        DESCRIPTION.
    A : TYPE, optional
        DESCRIPTION. The default is [].
    params : TYPE, optional
        DESCRIPTION. The default is {}.
    coefficients_similar_nets:  np.array [ # trials X # nets]
    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    A : TYPE
        DESCRIPTION.
    lambdas : np.array
        shape is [n_neurons X p]

    """
    if grannet:
        if not is_2d(coefficients_similar_nets):
            raise ValueError('coefficients_similar_nets should be 2d in dictionaryRWL1SF')
        if (checkEmptyList(coefficients_similar_nets) or trial_num < 0 or checkEmptyList(full_A)):
            raise ValueError('When calling for grannet, you must provide a non-negative trial number and coefficients for nets ')
        if trial_num > coefficients_similar_nets.shape[0]: 
            raise ValueError('The trial number cannot be larger than the number of trials in coefficients_similar_nets (shape of coefficients_similar_nets is) '+ str(coefficients_similar_nets.shape))
        
    # compute the presence coefficients from the dictionary
  
    params = {**{'epsilon': 1 , 'likely_from':'gaussian', 'numreps':2, 'normalizeSpatial':False,
                 'thresANullify': 2**(-50)},**params}
    if len(data.shape) == 3: data = MovToD2(data)
    n_times = data.shape[1]
    n_neurons = data.shape[0]
    p = phi.shape[1]
    
    corr_kern     = checkCorrKern(data, corr_kern); 
    
    """
    take sqrt of data
    """
    if params['to_sqrt']:        multi = np.sqrt(np.mean(data))
    else:        multi = 1
        
    """
    Check if A is empty
    """
    if checkEmptyList(A) or np.sum(A) == 0:
        if grannet and checkEmptyList(A):
            raise IndexErrorr('A should not be empty for grannet!')
        elif not grannet: 
            if params['dist_init'] == 'zeros':
                A = np.zeros((n_neurons, p))# np.zeros((n_neurons, p))
            elif  params['dist_init'] == 'rand':
                A = np.random.rand(n_neurons, p) * multi
            elif params['dist_init'] == 'uniform':
                A = np.random.uniform(0, 1,size = (n_neurons, p))*multi    
            elif params['dist_init'] == 'randn':
                A = np.random.randn(n_neurons, p) * multi
            else:
                raise ValueError('Unknown dist init') 
        else:
            pass
        
    if (isinstance( params['epsilon'] , list) and len(params['epsilon']) == 1):
        params['epsilon'] = params['epsilon'][0]
    if isinstance(params['epsilon'], numbers.Number):        
        lambdas = np.ones((n_neurons, p))*params['epsilon']
    elif (isinstance( params['epsilon'] , list) and len(params['epsilon']) == p):
        lambdas = np.repeat(params['epsilon'].reshape((1,-1)), n_neurons, axis = 0)#np.ones(n_neurons, p)*params['epsilon']
    elif (isinstance( params['epsilon'] , list) and len(params['epsilon']) == n_neurons):
        lambdas = np.repeat(params['epsilon'].reshape((-1,1)), p, axis = 1)#np.ones(n_neurons, p)*params['epsilon']
    else: 
        raise ValueError('Invalid length of params[epsilon]. Should be a number or a list with n_neurons or p elementts. Currently the params[epsilon] is ' + str(params['epsilon']))
    
   
    for repeat in range(params['numreps']):
        
        """
        Update the matrix of weights. [N X p] (if grannet - [N X p X trials])
        """
        if grannet:

            lambdas = updateLambdasMat(A, corr_kern, params['beta'], params)

        else:
            lambdas = updateLambdasMat(A, corr_kern, params['beta'], params)
        
        for n_neuron in range(n_neurons):
            if grannet:
                """
                Consider other nets
                """
                lambdas_vec = lambdas[n_neuron, :]
                #for trial_num in range(full_A.shape[2]):

                if params['likely_from'].lower() == 'gaussian': 

                    if data[n_neuron, :].mean() == 0:
                        full_A[n_neuron, :] = 0
                    else:
                        full_A[n_neuron, :, trial_num] = singleGaussNeuroInfer_grannet(data[n_neuron, :], phi, lambdas_vec[:, ], 
                                                                       full_A[n_neuron, :,:], trial_num, coefficients_similar_nets, 
                                                    params = params_default, l1 = params['l1'] , nonneg = params['nonneg'], 
                                                    ratio_null = 0.1, initial_round_free = initial_round_free)   
                elif params['likely_from'].lower() == 'poisson':
                    A[n_neuron, :] = singlePoiNeuroInfer_grannet(data[n_neuron, :], phi,lambdas_vec, full_A, 
                                                                 trial_num, coefficients_similar_nets, 
                                                params = params_default, l1 = params['l1'])
                else:
                    raise ValueError('Invalid likely from value')                

                
            else:

                """
                Do not consider other nets
                """
                if params['likely_from'].lower() == 'gaussian':               
                    A[n_neuron, :] = singleGaussNeuroInfer(lambdas[n_neuron, :], data[n_neuron, :],
                                                           phi, l1 = params['l1'], nonneg = params['nonneg'], 
                                                           A=A[n_neuron, :], params = params)

                elif params['likely_from'].lower() == 'poisson':

                    A[n_neuron, :] = singlePoiNeuroInfer(data[n_neuron, :],    phi,  lambdas[n_neuron, :],    params)
                else:
                    raise ValueError('Invalid likely from value')

    if params['normalizeSpatial']:
        max_A_over_neurons = A.sum(0)
        max_A_over_neurons[max_A_over_neurons == 0] = 1
        A = A/max_A_over_neurons.reshape((1,-1))
    A[A < params['thresANullify']] = 0

    if np.mean(A)  < 1e-29 and params['add_noise_small_A']:
        print('add noise!!')
        A = A + 0.05*np.random.randn(*A.shape)

    return A, lambdas

def singlePoiNeuroInfer(data, phi,lambdas_vec, params = params_default): 
    """  
    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    phi : TYPE
        DESCRIPTION.
    lambdas_vec : TYPE
        DESCRIPTION.
    params : TYPE, optional
        DESCRIPTION. The default is params_default.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    #THIS FUNCTION IS FOR UPDATING A
    phi_T_func = lambda x: phi.transpose().dot(x)
    phi_func = lambda x: phi.dot(x)
    finit = data.sum()*phi_T_func(data).size/phi_T_func(data).sum()/phi_T_func(np.ones_like(data)).sum() * phi_T_func(data)


    #raise ValueError('stop!')
    # pySPIRALTAP.    #SPIRALTAP()
    return SPIRALTAP(data, phi_func, lambdas_vec, AT = phi_T_func,Initialization = finit.reshape((-1,1)), 
                     noisetype =  'Poisson', **params['Poisson'])[0]


def is_1d(mat):
    if isinstance(mat,list): mat = np.array(mat)
    elif isinstance(mat, np.ndarray): pass
    else: raise ValueError('Mat must be numpy array or a list')
    return np.max(mat.shape) == len(mat.flatten())

def is_2d(mat, dim = 2):
    """
    Check if a matrix is 2-dimensional.

    Parameters
    ----------
    mat : list or np.ndarray
        The input matrix.
    dim : int, optional
        The number of dimensions to check for. The default is 2.

    Returns
    -------
    bool;         True if the matrix is 2-dimensional, False otherwise.

    Raises
    ------
    ValueError;         If `mat` is not a list or a `numpy` array.

    """    
    if isinstance(mat,list): mat = np.array(mat)
    elif isinstance(mat, np.ndarray): pass
    else: raise ValueError('Mat must be numpy array or a list')
    return (len(mat.shape) > dim and (np.array(mat.shape) != 1).sum() == dim) or (len(mat.shape) == dim and (1 not in mat.shape))

def singlePoiNeuroInfer_grannet(data, phi,lambdas_vec, full_A, trial_num, coefficients_similar_net, 
                                params = params_default, initial_round_free = True, l1 = 1):
    """
    THIS FUNCTION IS FOR UPDATING A in Grannet. 
    inv(phi) @ Y.T are concatinated vertically with all other networks to form least-square that consider all nets
    
    left_side = right_side @ A_(neuron_i, trial_i)
    
    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    phi : TYPE
        DESCRIPTION.
    lambdas_vec : TYPE
        DESCRIPTION.
    params : TYPE, optional
        DESCRIPTION. The default is params_default.
    full_A:
        if 2d: [p X trials]
        if 3d:[neurons X p X trials]
    coefficients_similar_nets_2d:   [# trials X p]  (originally full:   [# trials X # trials X # nets])
    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    # If initial round:
    if full_A[:,:,trial_num].sum() == 0 and initial_round_free:

        return singlePoiNeuroInfer(data, phi,lambdas_vec, params = params_default)
    
    # Else:
    if len(coefficients_similar_net.shape) == 3:
        coefficients_similar_nets_2d = coefficients_similar_nets[trial_num]
    else:
        coefficients_similar_nets_2d = coefficients_similar_nets
    coefficients_similar_nets_2d *= l1
    if (is_1d(data) and not is_2d(full_A)) or (not is_1d(data) and  is_2d(full_A)):
        raise ValueError('if data is 1d, full_A must be 2d. If full_A is 2d, data must be 1d')
    initialize_by_other_nets = params['grannet_params']['initialize_by_other_nets'] # True if to initialize A by the other nets
    include_Ai = params['grannet_params']['include_Ai']

    """
    create left side of equation
    """
    """ 3d case (all neurons together)"""
    if not is_2d(full_A): 
        # return a [trials*p X N]  vector
        # full A here is [N X p X trials]
        # each c here is [trials X p] matrix (elemment wise multipication with the relevant network)
        # full_A[:,:,counter] is [N X p]
        # coefficients_similar_nets_2d is [trials X p]
        # iterating over coefficients 
        nets_concat = np.vstack([np.repeat(c[trial_counter,:].reshape((-1,1)), full_A.shape[0], axis = 1)*full_A[:,:,trial_counter].T
                                 if (trial_counter != trial_num or include_Ai) 
                                 else  0*full_A[:,:,trial_counter]
                                for trial_, c in enumerate(coefficients_similar_nets_2d)])
        
    
    else:   
        """ 2d case (only one neuron) """
        # return a [trials*p X 1] vector
        # full_A is [p X trials]
        # coefficients_similar_nets_2d is [trials X p]
        # each c here is [# trials X p] vector (elemment wise multipication with the relevant network)
        nets_concat = np.vstack([(c.reshape((-1,1))*full_A[:,trial_counter].reshape((-1,1)) ).reshape((-1,1))
                                 if (trial_counter != trial_num or include_Ai) 
                                 else 0*full_A[:,counter].reshape((-1,1))
                                for trial_counter, c in enumerate(coefficients_similar_nets_2d)])
    """
    Add y 
    """
    if is_1d(data):
        # return a [T + trials X 1]
        data_concatinated_with_nets = np.vstack([ data.reshape((-1,1)), nets_concat])
    else:
        data_concatinated_with_nets = np.vstack([ data.T, nets_concat])    
        
    """
    Create right size of equation
    """
    # return [T + pXtrials X p]
    mult_A_concat = np.vstack([phi] + [np.eye(phi.shape[1]) if  (trial_counter != trial_num or include_Ai) else np.zeros((phi.shape[1],phi.shape[1])) 
                                  for trial_count in coefficients_similar_nets_2d.shape[0]])
    # Create a vector of coeffs [trials*p X 1] (trial 1, trial 2, etc.)
    #coefficients_similar_nets_2d_flatten = np.vstack([coefficients_similar_nets_2d_row.reshape((-1,1)) for
    #                                                  coefficients_similar_nets_2d_row in coefficients_similar_nets_2d])
    # if is_1d(data):

    #     # return a [T + trials X p]
    #     data_concatinated_with_nets = np.vstack([ , np.ones(coefficients_similar_nets_2d.shape[0],1) ])
    # else:
    #     np.re
    #data_concatinated_with_nets = np.vstack([ data.T, nets_concat])    
        
        
    #coeffs_with_default = np.vstack([1, coefficients_similar_nets.reshape((-1,1))])
    #if not include_Ai: coeffs_with_default[trial_num + 1] = 0
    #mult_A_concat =  np.vstack([coeff*np.eye(full_A.shape[1]) for coeff in coeffs_with_default])
    
    phi_T_func = lambda x: mult_A_concat.transpose().dot(x)
    phi_func = lambda x: mult_A_concat.dot(x)
    W = lambda x: np.diag(lambdas_vec).dot(x)
    WT = lambda x: np.diag(lambdas_vec).T.dot(x)
    
    """
    initialization
    """
    if initialize_by_other_nets: finit = []
    else:     
        finit = np.sum([coeff*full_A[:,:,counter] if (counter != trial_num or include_Ai) else 0*full_A[:,:,counter] 
                        for counter, coeff 
                        in enumerate(coefficients_similar_nets)])
        if  include_Ai:
            finit = finit/np.sum(coefficients_similar_nets)
        else:
            finit = finit/(np.sum(coefficients_similar_nets)-coefficients_similar_nets[trial_num])

    
    return pySPIRALTAP.SPIRALTAP(data_concatinated_with_nets, phi_func, lambdas_vec, 
                                 AT = phi_T_func, Initialization = finit,  
                                 W = W, WT = WT,   **params['Poisson'])[0]#penalty='onb', noooooooooooooooooo
    

def normalizeDictionary(D, cutoff = 1):
    D_norms = np.sqrt(np.sum(D**2,0))       # Get the norms of the dictionary elements 
    D       = D @ np.diag(1/(D_norms*(D_norms>cutoff)/cutoff+(D_norms<=cutoff))); # Re-normalize the basis
    return D

    
def dictionary_update(dict_old, A, data, step_s, GD_type = 'norm', params ={}):    
    if params['likely_from'].lower() == 'gaussian':
        dict_new = takeGDStep(dict_old, A, data, step_s, GD_type, params)
    if params['likely_from'].lower() == 'poisson':
        dict_new = takeGDStepPoisson(dict_old, A, data, step_s, GD_type, params)        
    if not params.get('normalizeSpatial'):
        dict_new = normalizeDictionary(dict_new,1)                            # Normalize the dictionary

    dict_new[np.isnan(dict_new)] = 0
    if np.mean(dict_new) < 1e-9:
        dict_new += np.random.rand(*dict_new.shape)
    return dict_new
    

def takeGDStep(dict_old, A, data, step_s, GD_type = 'norm', params ={}):
    """
    Parameters
    ----------
    dict_old : TYPE
        DESCRIPTION.
    A : TYPE
        DESCRIPTION.
    data : TYPE
        DESCRIPTION.
    step_s : TYPE
        DESCRIPTION.
    GD_type : TYPE, optional
        DESCRIPTION. The default is 'norm'.
    params : TYPE, optional
        DESCRIPTION. The default is {}.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    dict_new : TYPE
        DESCRIPTION.

    """
    l2 = params['l2'] # Frob. on dict
    l3 = params['l3'] # smoothness 
    l4 = params['l4'] # correaltions between dict elements
    
    if GD_type == 'norm':

        # Take a step in the negative gradient of the basis:
        # Minimizing the energy:    E = ||x-Da||_2^2 + lambda*||a||_1^2
        dict_new = update_GDiters(dict_old, A, data, step_s, params)

    elif GD_type == 'forb':
        # Take a step in the negative gradient of the basis:
        # This time the Forbenious norm is used to reduce unused
        # basis elements. The energy function being minimized is
        # then:     E = ||x-Da||_2^2 + lambda*||a||_1^2 + lamForb||D||_F^2
        dict_new = update_GDwithForb(dict_old, A, data, step_s, l2, params);
    elif GD_type ==  'full_ls':
        # Minimizing the energy:
        # E = ||X-DA||_2^2 via D = X*pinv(A)
        dict_new = update_FullLS(dict_old, A, data, params);
    elif GD_type == 'anchor_ls':
        # Minimizing the energy:
        # E = ||X-DA||_2^2 + lamCont*||D_old - D||_F^2 via D = [X;D_old]*pinv([A;I])
        dict_new = update_LSwithCont(dict_old, A, data, l3, params);
    elif GD_type == 'anchor_ls_forb':
        # Minimizing the energy:
        # E = ||X-DA||_2^2 + lamCont*||D_old - D||_F^2 + lamForb*||D||_F^2 
        #                  via D = [X;D_old]*pinv([A;I])
        dict_new = update_LSwithContForb(dict_old, A, data, l2, l3, params);
    elif GD_type == 'full_ls_forb':
        # Minimizing the energy:
        # E = ||X-DA||_2^2 + lamForb*||D||_F^2
        #              via  D = X*A^T*pinv(AA^T+lamForb*I)
        dict_new = update_LSwithForb(dict_old, A, data, l2, params);
    elif GD_type== 'full_ls_cor':
        # E = ||X-DA||_2^2 + l4*||D.'D-diag(D.'D)||_sav + l2*||D||_F^2
        #             + l3*||D-Dold||_F^2 
        dict_new = update_FullLsCor(dict_old, A, data, l2, l3, l4, params)
    elif GD_type =='sparse_deconv':
        dict_new   = sparseDeconvDictEst(dict_old,data,A,params.h,params); # This is a more involved function and needs its own function
    else:
        raise ValueError('GD_Type %s is not defined in the takeGDstep function'%GD_type)        

    return dict_new
    
    
    
    
    
def takeGDStepPoisson(dict_old, A, data, step_s, GD_type = 'norm', params ={}):
    """
    Parameters
    ----------
    dict_old : TYPE
        DESCRIPTION.
    A : TYPE
        DESCRIPTION.
    data : TYPE
        DESCRIPTION.
    step_s : TYPE
        DESCRIPTION.
    GD_type : TYPE, optional
        DESCRIPTION. The default is 'norm'.
    params : TYPE, optional
        DESCRIPTION. The default is {}.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    dict_new : TYPE
        DESCRIPTION.

    """
    l2 = params['l2']
    l3 = params['l3']
    l4 = params['l4']
    
    if GD_type == 'norm':


        # Take a step in the negative gradient of the basis:
        # Minimizing the energy:    E = ||x-Da||_2^2 + lambda*||a||_1^2
        dict_new = update_GDiters_Poisson(dict_old, A, data, step_s, params)

        
    elif GD_type == 'forb':
        # Take a step in the negative gradient of the basis:
        # This time the Forbenious norm is used to reduce unused
        # basis elements. The energy function being minimized is
        # then:     E = ||x-Da||_2^2 + lambda*||a||_1^2 + lamForb||D||_F^2
        dict_new = update_GDwithForb_Poisson(dict_old, A, data, step_s, l2, params);
    elif GD_type ==  'full_ls':
        # Minimizing the energy:
        # E = ||X-DA||_2^2 via D = X*pinv(A)
        dict_new = update_FullLS_Poisson(dict_old, A, data, params);
    elif GD_type == 'anchor_ls':
        # Minimizing the energy:
        # E = ||X-DA||_2^2 + lamCont*||D_old - D||_F^2 via D = [X;D_old]*pinv([A;I])
        dict_new = update_LSwithCont_Poisson(dict_old, A, data, l3, params);
    elif GD_type == 'anchor_ls_forb':
        # Minimizing the energy:
        # E = ||X-DA||_2^2 + lamCont*||D_old - D||_F^2 + lamForb*||D||_F^2 
        #                  via D = [X;D_old]*pinv([A;I])
        dict_new = update_LSwithContForb_Poisson(dict_old, A, data, l2, l3, params, step_s);
    elif GD_type == 'full_ls_forb':
        # Minimizing the energy:
        # E = ||X-DA||_2^2 + l2*||D||_F^2
        #              via  D = X*A^T*pinv(AA^T+lamForb*I)
        dict_new = update_LSwithForb_Poisson(dict_old, A, data, l2, params);
    elif GD_type== 'full_ls_cor':
        # E = ||X-DA||_2^2 + l4*||D.'D-diag(D.'D)||_sav + l2*||D||_F^2
        #             + l3*||D-Dold||_F^2 
        dict_new = full_ls_cor_poisson(dict_old, A, data, l2, l3, l4, params, step_s = step_s)
    elif GD_type =='sparse_deconv':
        dict_new   = sparseDeconvDictEst_Poisson(dict_old,data,A,params.h,params); # This is a more involved function and needs its own function
    else:
        raise ValueError('GD_Type %s is not defined in the takeGDstep function'%GD_type)        

    return dict_new    
    
    
    
    
    
    
    
    
def update_GDwithForb_Poisson(dict_old, A, data, step_s, l2, params):
    """
    Take a step in the negative gradient of the basis:
    This time the Forbenious norm is used to reduce unused basis elements. The energy function being minimized is then:
     E = ||denomi - ylog(denomi)||_2^2 
    
    Parameters
    ----------
    dict_old : TYPE
    A : TYPE
    data : TYPE
    step_s : TYPE
        DESCRIPTION.
    l2 : TYPE
        DESCRIPTION.
    params : TYPE
        DESCRIPTION.

    Returns - new dict
    -------
    """

    for index2 in range(params.get('GD_iters')):
        # Update The basis matrix
        denomi = (dict_old @ A.T).T;
        grad_matrix = gradient_log(A, data, denomi);
        #dict_new = dict_old + (step_s)*((data.T - dict_old @ A.T) @ A -l2*dict_old) @ np.diag(1/(1+np.sum(A != 0, 0)));
        no_reg_term = (step_s/A.shape[0])*(-grad_matrix + np.ones((data.shape[1],1)) @ np.ones((1, A.shape[0])) @ A)
        reg_term = (step_s/A.shape[0])*(-l2*dict_old)*np.diag(1/(1+np.sum(A != 0, 1)))
        dict_old = dict_old - no_reg_term + reg_term
        
    
        # For some data sets, the basis needs to be non-neg as well
        if params.get('nonneg'):
            dict_new[dict_new < 0] = 0 + epsilon
    return dict_new
      
    
    
    
    
    
    
    
    
    
    
    
def dictInitialize(phi = [], shape_dict = [], norm_type = 'unit', to_norm = True, params = {}):

    """
    Parameters
    ----------
    phi : list of lists or numpy array or empty list
        The initializaed dictionary
    shape_dict : tuple or numpy array or list, 2 int elements, optional
        shape of the dictionary. The default is [].
    norm_type : TYPE, optional
        DESCRIPTION. The default is 'unit'.
    to_norm : TYPE, optional
        DESCRIPTION. The default is True.
    dist : string, optional
        distribution from which the dictionary is drawn. The default is 'uniforrm'.        
    Raises
    ------
    ValueError
        DESCRIPTION.
        
    Returns
    -------
    phi : TYPE
        The output dictionary

    """

    if len(phi) == 0 and len(shape_dict) == 0:
        raise ValueError('At least one of "phi" or "shape_dict" must not be empty!')
    if len(phi) > 0:        
        return norm_mat(phi, type_norm = norm_type, to_norm = to_norm)
    else:
        #if dist == 'uniform':
        phi = createMat(shape_dict, params)


        return dictInitialize(phi, shape_dict, norm_type, to_norm,  params)

    
def createMat(shape_dict,  params = params_default ):
    """
    Parameters
    ----------
    shape : TYPE
        DESCRIPTION.
    dist : TYPE, optional
        DESCRIPTION. The default is 'uniforrm'.
    params : TYPE, optional
        DESCRIPTION. The default is {'mu':0, 'std': 1}.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
    """

    params = {**{'mu':0, 'std': 1}, **params}
    dist = params['dist_init']

    if dist == 'uniform':
        return np.random.uniform(params[uniform_vals[0]], params[uniform_vals[1]], size = (shape_dict[0], shape_dict[1]) )
    elif dist == 'rand':
        return np.random.rand(shape_dict[0], shape_dict[1]) 
    elif dist == 'randn':
        return np.random.randn(shape_dict[0], shape_dict[1]) 
    elif dist == 'norm':
        return params['mu'] + np.random.randn(shape_dict[0], shape_dict[1])*params['std']
    elif dist == 'zeros':
        return np.zeros((shape_dict[0], shape_dict[1]))
    else:
        raise ValueError('Unknown dist for createMat')
#<h1 id="header">Header</h1>    

def is_pos_def(x):

    return np.all(np.linalg.eigvals(x) > 0)


def singleGaussNeuroInfer(lambdas_vec, data, phi, l1,  nonneg, A = [], 
                          ratio_null = 0.1, params = {}, grannet = False):
    # Use quadprog  to solve the weighted LASSO problem for a single vector
    #include_Ai = params['grannet_params']['include_Ai']
    if phi.shape[1] != len(lambdas_vec):
        raise ValueError('Dimension mismatch!')        
    # Set up problem
    data = data.flatten()                                           # Make sure time-trace is a column vector
    lambdas_vec = lambdas_vec.flatten()                             # Make sure weight vector is a column vector
    p      = len(lambdas_vec)                                       # Get the numner of dictionary atoms
                                                                    ## Run the weighted LASSO to get the coefficients    
    if len(data) == 0 or np.sum(data**2) == 0:
        A = np.zeros(p)                                             # This is the trivial solution to generate all zeros linearly.
        raise ValueError('zeros again')
    else:
        if nonneg:
            if A == [] or (A==0).all():
                A = solve_qp(2*(phi.T @ phi) , -2*phi.T @ data + l1*lambdas_vec, 
                             solver = params['solver_qp'] )       # Use quadratic programming to solve the non-negative LASSO
                if np.nan in A: raise ValueError('nan')
                ub = np.inf*np.ones((p,1)),
            else:

                if (not is_pos_def(phi.T @ phi)) and  (params['deal_nonneg'] == 'make_nonneg'):
                    phi_T_phi = phi.T @ phi + epsilon
                elif not is_pos_def(phi.T @ phi):
                    A = solve_Lasso_style(phi, data, l1, [], params = params, random_state = 0).flatten()
                    #solver_L1RLS(phi, data, l1, zeros(N2, 1), params )         # Solve the weighted LASSO using TFOCS and a modified linear operator
                    if params['norm_by_lambdas_vec']:
                        A = A.flatten()/lambdas_vec.flatten();              # Re-normalize to get weighted LASSO values
                        #  consider changing to oscar like here https://github.com/vene/pyowl/blob/master/pyowl.py 
                else:
                    phi_T_phi = phi.T @ phi
                    
                A = solve_qp(2*(phi.T @ phi),-2*phi.T @ data+l1*lambdas_vec, 
                             solver = params['solver_qp'] )         # Use quadratic programming to solve the non-negative LASSO

                if np.isnan(A).any(): 
                    raise ValueError('There are nan values is A')
          
        else:
           A = solve_Lasso_style(phi, data, l1, [], params = params, random_state = 0).flatten()
           #solver_L1RLS(phi, data, l1, zeros(N2, 1), params )         # Solve the weighted LASSO using TFOCS and a modified linear operator
           if params['norm_by_lambdas_vec']:
               A = A.flatten()/lambdas_vec.flatten();              # Re-normalize to get weighted LASSO values
               #  consider changing to oscar like here https://github.com/vene/pyowl/blob/master/pyowl.py 
    if params['nullify_some']:
        A[A<ratio_null*np.max(A)] = 0;    
    return A


def singleGaussNeuroInfer_grannet(data, phi, lambdas_vec, full_A_2d, trial_num, 
                                  coefficients_similar_nets,  params = params_default, 
                        l1 = 1,  nonneg = False, ratio_null = 0.1, initial_round_free = True):
    """
    Use quadprog  to solve the weighted LASSO problem for a single vector
    Update A by lasso.
    Available only in 2d
    
    Parameters
    ----------
    lambdas_vec : vector with p elements.
        DESCRIPTION.
    data : TYPE
        DESCRIPTION.
    phi : TYPE
        DESCRIPTION.
    l1 : TYPE
        DESCRIPTION.
    nonneg : TYPE
        DESCRIPTION.
    A : TYPE, optional
        DESCRIPTION. The default is [].
    ratio_null : TYPE, optional
        DESCRIPTION. The default is 0.1.
    params : TYPE, optional
        DESCRIPTION. The default is {}.
    grannet : TYPE, optional
        DESCRIPTION. The default is False.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    A : TYPE
        DESCRIPTION.

    """  
    include_Ai = params['grannet_params']['include_Ai']
    labels_indicative = params['grannet_params']['labels_indicative']
    if full_A_2d[:,trial_num].sum() == 0 and initial_round_free:
        return singleGaussNeuroInfer(lambdas_vec, data, phi, l1,  nonneg, A = full_A_2d[:,trial_num], 
                                  ratio_null =  ratio_null, params = params)
    if labels_indicative:
        if len(coefficients_similar_nets.shape) == 3:
            coefficients_similar_nets_2d = coefficients_similar_nets[trial_num] # trial num X num nets
        else:
            coefficients_similar_nets_2d = coefficients_similar_nets
    else:
        coefficients_similar_nets_2d = np.ones(full_A_2d.shape).T
        
    # check if lambas_vec has p elements
    if phi.shape[1] != len(lambdas_vec.flatten()):        raise ValueError('Dimension mismatch!')   
    p   = phi.shape[1]                                    # Get the numner of dictionary atoms     

                                                                    ## Run the weighted LASSO to get the coefficients    
    if len(data) == 0 or np.sum(data**2) == 0:        
        raise ValueError('zeros again')
        
    else:

        new_data = np.vstack([data.reshape((-1,1)), np.vstack([(coefficients_similar_nets_2d[trial_counter,:].reshape((-1,1)) * full_A_2d[:,trial_counter].reshape((-1,1)) ).reshape((-1,1))
                                 if (trial_counter != trial_num or include_Ai) 
                                 else 0*full_A_2d[:,trial_counter].reshape((-1,1))
                                for trial_counter in range(coefficients_similar_nets_2d.shape[0])])]).reshape((-1,1))
        new_phi = np.vstack([phi] + [np.eye(phi.shape[1]) if  (trial_counter != trial_num or include_Ai) else np.zeros((phi.shape[1],phi.shape[1])) 
                                      for trial_counter in range(coefficients_similar_nets_2d.shape[0])])
        
        if nonneg:
            A = solve_qp(2*(new_phi.T @ new_phi),-2*new_phi.T @ new_data + l1*lambdas_vec, 
                         solver = params['solver_qp'] ,   initvals = full_A_2d[:,trial_num])         # Use quadratic programming to solve the non-negative LASSO

            if np.isnan(A).any(): 
                raise ValueError('There are nan values is A')
            full_A_2d[:,trial_num] = A
      
        else:
           A = solve_Lasso_style(new_phi, new_data, l1, full_A_2d[:,trial_num], 
                                 params = params, random_state = 0).flatten()
           #solver_L1RLS(phi, data, l1, zeros(N2, 1), params )         # Solve the weighted LASSO using TFOCS and a modified linear operator
           if params['norm_by_lambdas_vec']:
               A = A.flatten()/lambdas_vec.flatten();              # Re-normalize to get weighted LASSO values
               #  consider changing to oscar like here https://github.com/vene/pyowl/blob/master/pyowl.py 
           full_A_2d[:,trial_num] = A
    if params['nullify_some']:
        full_A_2d[:,trial_num][full_A_2d[:,trial_num] < ratio_null*np.max(full_A_2d[:,trial_num])] = 0;    
    return full_A_2d[:,trial_num]


def solve_Lasso_style(A, b, l1, x0, params = {}, lasso_params = {},random_state = 0):
  """
      Solves the l1-regularized least squares problem
          minimize (1/2)*norm( A * x - b )^2 + l1 * norm( x, 1 ) 
          
    Parameters
    ----------
    A : TYPE
        DESCRIPTION.
    b : TYPE
        DESCRIPTION.
    l1 : float
        scalar between 0 to 1, describe the reg. term on the cofficients.
    x0 : TYPE
        DESCRIPTION.
    params : TYPE, optional
        DESCRIPTION. The default is {}.
    lasso_params : TYPE, optional
        DESCRIPTION. The default is {}.
    random_state : int, optional
        random state for reproducability. The default is 0.

    Raises
    ------
    NameError
        DESCRIPTION.

    Returns
    -------
    x : np.ndarray
        the solution for min (1/2)*norm( A * x - b )^2 + l1 * norm( x, 1 ) .

  lasso_options:
               - 'inv' (least squares)
               - 'lasso' (sklearn lasso)
               - 'fista' (https://pylops.readthedocs.io/en/latest/api/generated/pylops.optimization.sparsity.FISTA.html)
               - 'omp' (https://pylops.readthedocs.io/en/latest/gallery/plot_ista.html#sphx-glr-gallery-plot-ista-py)
               - 'ista' (https://pylops.readthedocs.io/en/latest/api/generated/pylops.optimization.sparsity.ISTA.html)       
               - 'IRLS' (https://pylops.readthedocs.io/en/latest/api/generated/pylops.optimization.sparsity.IRLS.html)
               - 'spgl1' (https://pylops.readthedocs.io/en/latest/api/generated/pylops.optimization.sparsity.SPGL1.html)
               
               
               - . Refers to the way the coefficients should be claculated (inv -> no l1 regularization)
  """  
  if len(b.flatten()) == np.max(b.shape):
      b = b.reshape((-1,1))
  if 'solver' not in params.keys():
      warnings.warn('Pay Attention: Using Default (inv) solver for updating A. If you want to use lasso please change the solver key in params to lasso or another option from "solve_Lasso_style"')
  params = {**{'threshkind':'soft','solver':'inv','num_iters':50}, **params}

  if params['solver'] == 'inv' or l1 == 0:

      x =linalg.pinv(A) @ b.reshape((-1,1))

  elif params['solver'] == 'lasso' :
      #fixing try without warm start
    clf = linear_model.Lasso(alpha=l1,random_state=random_state, **lasso_params)
    # print('b shape')
    # print(b.shape)
    # print('A shape')
    # print(A.shape)
    if np.max(b.shape) == len(b.flatten()):
        x = clf.fit(A,b.reshape((-1,1)) )     
    else:
        x = clf.fit(A,b.T )     
    x = np.array(clf.coef_)

  elif params['solver'].lower() == 'fista' :
      Aop = pylops.MatrixMult(A)
  
      #if 'threshkind' not in params: params['threshkind'] ='soft'
      #other_params = {'':other_params[''],
      x = pylops.optimization.sparsity.FISTA(Aop, b.flatten(), niter=params['num_iters'],
                                             eps = l1 , threshkind =  params.get('threshkind') )[0]
  elif params['solver'].lower() == 'ista' :

      #fixing try without warm start
      if 'threshkind' not in params: params['threshkind'] ='soft'
      Aop = pylops.MatrixMult(A)
      x = pylops.optimization.sparsity.ISTA(Aop, b.flatten(), niter=params['num_iters'] , 
                                                 eps = l1,threshkind =  params.get('threshkind'))[0]
      
  elif params['solver'].lower() == 'omp' :
  
      Aop = pylops.MatrixMult(A)
      x  = pylops.optimization.sparsity.OMP(Aop, b.flatten(), 
                                                 niter_outer=params['num_iters'], sigma=l1)[0]     
  elif params['solver'].lower() == 'spgl1' :
    
      Aop = pylops.MatrixMult(A)
      x = pylops.optimization.sparsity.SPGL1(Aop, b.flatten(),iter_lim = params['num_iters'],  tau = l1)[0]      
      
  elif params['solver'].lower() == 'irls' :
   
      Aop = pylops.MatrixMult(A)
      
      #fixing try without warm start
      x = pylops.optimization.sparsity.IRLS(Aop, b.flatten(),  nouter=50, espI = l1)[0]      
  else:     
    raise NameError('Unknown update c type')  
  return x


def updateLambdasMat(A, corr_kern, beta, params ):

    p = A.shape[1]
    n_neurons = A.shape[0]
    params = {**{'epsilon':1, 'updateEmbed': False, 'mask':[]}, **params}
    if params.get('updateEmbed')  :                    # If required, recalculate the graph based on the current estimate of coefficients
        raise ValueError('not implemented yet!')
        # H = mkDataGraph(A, []);   
                                            # This line actually runs that re-calculation of the graph
    if (isinstance( params['epsilon'] , list) and len(params['epsilon']) == 1):
        params['epsilon'] = params['epsilon'][0]
        
    if isinstance(params['epsilon'], numbers.Number):     # If the numerator of the weight updates is constant...
        
        if params['updateEmbed']  :       #  - If the numerator of the weight updates is the size of the dictionary (i.e., one tau per dictioary element)...
            # create the lambdas of grannet   
            lambdas = params['epsilon']/(beta + A + H @ A)                            #    - Calculate the wright updates tau/(beta + |s_i| + [P*S]_i) - calculate lambda!!!!
        elif not params['updateEmbed']:     
                                                #  - If the graph was not updated, use the original graph (in corr_kern)
            if corr_kern.shape[0] ==  n_neurons    :                                 #    - If the weight projection matrix has the same number of rows as pixels, update based on a matrix multiplication                     

                lambdas = params['epsilon']/(beta + A + corr_kern @ A);                #      - Calculate the wright updates tau/(beta + |s_i| + [P*S]_i)
            else:
                raise ValueError('This case is not defined yet') #future
                # CF = np.zeros(A.shape);
                # for net_num in range(p):
                #     if not params.get('mask')
                #         CF[:,net-num] = np.convolve(conv2(reshape(S(:,i),params.nRows,params.nCols), corr_kern, mode = 'same');
                #     else
                #         temp = zeros(params.nRows,params.nCols);
                #         temp(params.mask(:)) = S(:,i);
                #         temp = conv2(temp ,corr_kern,'same');
                #         CF(:,i) = temp(params.mask(:));

                # lambdas = params['epsilon']/(beta + A + CF);  #      - Calculate the wright updates tau/(beta + |s_i| + [P*S]_i)

    elif len(params['epsilon'].flatten()) == p:            # If the numerator of the weight updates is the size of the dictionary (i.e., one tau per dictioary element)...
   
        if params['updateEmbed'] :                                            #  - If the graph was updated, use the new graph (i.e., P)
            lambdas = params['epsilon'].reshape((1,-1))/(beta + A + corr_kern @ A)         #    - Calculate the wright updates tau/(beta + |s_i| + [P*S]_i)
        else   :                                       #  - If the graph was not updated, use the original graph (in corr_kern)
            if corr_kern.shape[0] == n_neurons :                                    #    - If the weight projection matrix has the same number of rows as pixels, update based on a matrix multiplication
                lambdas =  params['epsilon'].reshape((1,-1))/(beta + A + corr_kern @ A) #    - Calculate the wright updates tau/(beta + |s_i| + [P*S]_i)
            else   :
                raise ValueError('Invalid kernel shape') #future                #  - Otherwise, the graph was updated; use the original graph (in corr_kern)
            # I'm not sure what option is supposed to go here
            #    lambdas = reshape(convn(reshape(S, [im_x, im_y, nd]),...
            #                                  corr_kern,'same'), im_x*im_y,1); #    - Get base weights
            #    lambdas = bsxfun(@times, params['epsilon'], 1./(beta + S + lambdas)); #    - Calculate the wright updates tau/(beta + |s_i| + [P*S]_i) 

    elif params['epsilon'].shape[0] == A.shape[0] and params['epsilon'].shape[1] == A.shape[1]: #future
    
        raise ValueError('This option is not available yet')
        # If the numerator of the weight updates is the size of the image
        #CF = np.zeros(A.shape)
        #for net_num in range(A.shape[1]): #future
            #if not params.get('mask')
            #    CF(:,i) = vec(conv2(reshape(S(:,i),params.nRows,params.nCols)  ,corr_kern,'same'));
            #else
            #         temp = zeros(params.nRows,params.nCols);
            #         temp(params.mask(:)) = S(:,i);
            #         CF(:,i) = vec(conv2(temp ,corr_kern,'same'));
        
            # lambdas = bsxfun(@times, params['epsilon'], ones(1,1,nd))./(beta+S+CF);  %  - Calculate the wright updates tau/(beta + |s_i| + [P*S]_i)
            
    else:
        raise ValueError('Invalid Option')
    return lambdas
    
def MovToD2(mov):
    """
    PAY ATTENTION! THIS ONE IS NOT FOR GRANNET. THE EXPECTATION IS  [pixels X pixels X time]
    Parameters
    ----------
    mov : can be list of np.ndarray of frames OR 3d np.ndarray of [pixels X pixels X time]
        The data

    Returns
    -------
    array 
        a 2d numpy array of the movie, pixels X time 

    """
    if isinstance(mov, list):
        return np.hstack([frame.flatten().reshape((-1,1)) for frame in mov])
    elif isinstance(mov, np.ndarray) and len(np.shape(mov)) == 2:
        return mov
        #elif isinstance(mov, np.ndarray) and len(np.shape(mov)) == 2:
        #    return np.hstack([mov[:,:,frame_num].flatten().reshape((-1,1)) for frame_num in range(mov.shape[2])])
    elif isinstance(mov, np.ndarray) and len(np.shape(mov)) == 3:
        print('start calculated movtod2')
        to_d2_return = np.hstack([mov[:,:,frame_num].flatten().reshape((-1,1)) for frame_num in range(mov.shape[2])])  
        print('end calculated movtod2')
        return   to_d2_return
    else:
        raise ValueError('Unrecognized dimensions for mov (cannot change its dimensions to 2d)')
    
def MovToD2_grannet(data): 
    # data 3d is expected to be N X T x k
    # return MovToD2(np.transpose(data_3d,(2,1,0))).T #np.transpose(, (2,1,0))
    return np.hstack([data[:,:,k] for k in range(data.shape[2])])
    
def order_A_results(full_A, full_phi):
    # order A2 according to A1
    num_trials = int(full_phi.shape[2] / full_A.shape[2])
    full_A_new = [full_A[:,:,0]]
    full_phi_new = [full_phi[:,:,:num_trials]]
    for trial in range(1,full_A.shape[2]):
        A_order_T, _, list_reorder  = match_times(full_A_new[-1].T, full_A[:,:,trial].T, 
                                                  full_phi[:,:,trial*num_trials], 
                                                  enable_different_size = False, add_low_corr = False )
        cur_phi = full_phi[:,:,trial*num_trials:num_trials*(trial+1)]
        phi_order_T = np.dstack([cur_phi[:,list_reorder,phi_num]  for phi_num in range(cur_phi.shape[2])   ])
        full_A_new.append(A_order_T.T)
        full_phi_new.append(phi_order_T)
    
    full_A_new = np.dstack(full_A_new)
    full_phi_new = np.dstack(full_phi_new)
    return full_A_new, full_phi_new
    
    
    
    
    
    
def D2ToMov(mov, frameShape, type_return = 'array'):
    """
    Parameters
    ----------
    mov : TYPE
        DESCRIPTION.
    frameShape : TYPE
        DESCRIPTION.
    type_return : string, can be 'array' or 'list', optional
        The default is 'array'.
        
    Raises
    ------
    ValueError - if dimensions do not fit


    Returns
    -------
    list or np.ndarray (according to the input "type return") of frames with shape frameShape X time
    """
    
    if mov.shape[0] != frameShape[0]*frameShape[1] :
        raise ValueError('Shape of each frame ("frameShape") is not consistent with the length of the data ("mov")')
    if type_return == 'array':
        return np.dstack([mov[:,frame].reshape(frameShape) for frame in range(mov.shape[1])])
    elif type_return == 'list':     
        return [mov[:,frame].reshape(frameShape) for frame in range(mov.shape[1])]
    else:
        raise ValueError('Invalid "type_return" input. Should be "list" or "array"')
    
def snythetic_evaluation(full_A, full_phi, real_full_A = {}, real_full_phi = {}):
    #     given ground truth  dynamics - compare the results
    #d_full = fnp.load(r'grannet_synth_results_march_2023.npy', allow_pickle=True).item()
    if len(real_full_A ) == 0 or len(real_full_phi) == 0:
        d_full = np.load(r'grannet_synth_results_march_2023.npy', allow_pickle=True).item()
        real_full_A = d_full['A']
        real_full_phi = d_full['phi']
        
    ordered_A = []
    ordered_phi = []
    for counter, (cond, phi1) in enumerate(real_full_phi.items()):     
        phi2, A, _ = match_times(phi1, full_phi[:,:,counter].T, full_A[:,:,counter]) #for counter, (cond, phi1) in enumerate(real_full_phi)
        ordered_phi.append(phi2)
        ordered_A.append(A)
    full_phi_ordered = np.dstack(ordered_phi)
    full_A_ordered = np.dstack(ordered_A)
    
    return full_phi_ordered, full_A_ordered
    
def spec_corr(v1,v2, to_abs = True):
  """
  absolute value of correlation
  """
  corr = np.corrcoef(v1.flatten(),v2.flatten())
  if to_abs:
      return np.abs(corr[0,1])
  return corr[0,1]
    
#def order_A_tog(A_3d, phi_3d):
    
    
    
    
    
def match_times(phi1, phi2, A, enable_different_size = True, add_low_corr = False ):
    """
    Reorders nets by times and changes phi2 and A2 according to phi1.

    Parameters:
    -----------
    phi1 : array_like
        2D array of shape (p, T) representing the first set of signals.
    phi2 : array_like
        2D array of shape (p, T) representing the second set of signals.
    A : array_like
        2D array of shape (N, p) representing the linear filters.
    enable_different_size : bool, optional
        Whether to enable the input arrays to have different shapes. The default is True.
    add_low_corr : bool, optional
        Whether to add low correlation nets. The default is False.

    Returns:
    --------
    phi2 : array_like
        2D array of shape (p, T) representing the reordered second set of signals.
    A : array_like
        2D array of shape (N, p) representing the reordered linear filters.
    list_reorder : array_like
        1D array of length p representing the order in which the signals were reordered.

    reorder nets by times
    changing phi2 and A2 according to phi1
    


    """
    turnphi = False
    turnA = False
    
    if phi1.shape != phi2.shape and not enable_different_size:
        raise ValueError('phi1 and phi2 must have the same shape but shape(phi1) is %s and shape of phi2 is %s'%(phi1.shape,phi2.shape))
    
    if phi1.shape[0] != A.shape[1]:
        print('change direction of matrices')
        if phi1.shape[0] == A.shape[0]: # here A is not ok
            A = A.T
            turnA = True
        elif phi1.shape[1] == A.shape[1]: # here phi is not ok
            phi = phi.T
            turnphi = True
        elif  phi1.shape[1] == A.shape[0]: # here A is not ok
            A = A.T
            phi = phi.T
            turnphi = True
            turnA = True
            
    p = A.shape[1]
    """
    rows = of phi1
    cols - of phi2
    """
    corr_mat = np.zeros((phi1.shape[0],phi2.shape[0]))
    for p_spec1 in range(phi1.shape[0]):
        for p_spec2 in range(phi2.shape[0]):
            corr_mat[p_spec1, p_spec2] = spec_corr(phi1[p_spec1], phi2[p_spec2], to_abs = False)
    
    list_reorder = np.zeros(p)
    list_sign = np.zeros(p)
    for _ in range(p):
        
        inds = np.unravel_index( np.argmax(np.abs(corr_mat))  , corr_mat.shape )
      
        list_reorder[inds[0]] = inds[1].astype(int)
        if corr_mat[inds[0],inds[1]] >= 0:
            list_sign[inds[0]] = 1
        else:
            list_sign[inds[0]] =-1
        corr_mat[inds[0],:] = 0 #np.delete(corr_mat, inds[0], axis = 0 )
        corr_mat[:,inds[1]] = 0
        #corr_mat = #np.delete(corr_mat, inds[1], axis = 1 )
    
    if phi1.shape[0] > phi2.shape[0] :
        add_redundant = False # not a problem here #not_included = [el for el in np.arange(phi1.shape[0]) if el not in list_reorder]
    elif phi1.shape[0] < phi2.shape[0] : # more nets by grannet
        not_included = np.array([el for el in np.arange(phi2.shape[0]) if el not in list_reorder])
        phi2_redu = phi2[not_included,:]
        A_redu = A[:,not_included]
        add_redundant = True
    else:
        add_redundant = False
    list_reorder = list_reorder.astype(int)
    
    
    phi2 = phi2[list_reorder,:]
    phi2 = np.vstack([phi2[p,:].reshape((1,-1))*list_sign[p] for p in range(phi2.shape[0])])
    
    A = A[:,list_reorder]
    A = np.hstack([A[:,p].reshape((-1,1))*list_sign[p] for p in range(A.shape[1])])
    if add_redundant and add_low_corr:
        phi2 = np.vstack([phi2, phi2_redu])
        A = np.hstack([A, A_redu])
        
    if turnphi:
        phi2 = phi2.T
    if turnA:
        A = A.T
    return phi2, A, list_reorder 
        
    
            
            
            # add threhold to A!!!
            # add normalization!! 
    
    

def mkDataGraph(data, params = {}, reduceDim = False, reduceDimParams = {}, graph_function = 'gaussian',
                K_sym  = True, use_former = False, data_name = 'none', toNormRows = True,
                graph_params = params_default['graph_params'],
                grannet = False):
    """
    Parameters
    ----------
    data : should be neurons X time OR neurons X p
        DESCRIPTION.
    params : TYPE, optional
        DESCRIPTION. The default is {}.
    reduceDim : TYPE, optional
        DESCRIPTION. The default is False.
    reduceDimParams : TYPE, optional
        DESCRIPTION. The default is {}.
    graph_function : TYPE, optional
        DESCRIPTION. The default is 'gaussian'.
    K_sym : TYPE, optional
        DESCRIPTION. The default is True.
    use_former : TYPE, optional
        DESCRIPTION. The default is True.
    data_name : TYPE, optional
        DESCRIPTION. The default is 'none'.
    toNormRows : TYPE, optional
        DESCRIPTION. The default is True.
    data - 3d case, needed only for grannet
    graph_params - paramgs_graph:
        'kernel_grannet_type' can be 'ind' or "averaged"  or "combination"  or 'one_kernel'

    Returns
    -------
    TYPE
        DESCRIPTION.
    """
    
    #if grannet and graph_params['kernel_grannet_type'] != 'ind' and checkEmptyList(data):
    #    raise ValueError('you must provide data for non-ind graph! but grannet type is %s'%graph_params['kernel_grannet_type'])
    if not grannet or len(data.shape) == 2:# (grannet and graph_params['kernel_grannet_type'] == 'ind'):  # and checkEmptyList(data):
        """
        IN THIS CASE IT CALCULATES THE KERNEL AND RETURN A SPARSE MATRIX WITH VALUES ONLY IN THE KNN
        """
        reduceDimParams = {**{'alg':'PCA'},  **reduceDimParams}
        params = addKeyToDict(params_config,
                     params)
        if len(data.shape) == 3:
            data = np.hstack([data[:,:,i].flatten().reshape((-1,1)) for i in range(data.shape[2])])
            print('data was reshaped to 2d')
            # Future: PCA
        if reduceDim:
            pca = PCA(n_components=params['n_comps'])
            data = pca.fit_transform(data)
    
        K = calcAffinityMat(data, params,  data_name, use_former, K_sym, graph_function, 
                             graph_params = graph_params,   grannet = grannet)   
        K = K - np.diag(np.diag(K) ) 
    
        if toNormRows:
            K = K/K.sum(1).reshape((-1,1))
    else: # in case of grannet
        """
        IN THIS CASE IT CALCULATES THE distances between 2 neurons
        """
        raise ValueError('how did you arrive here? if grannet and not ind type, you should use the grannet function (mkDataGraph_grannet), not this one!')
        
        # if len(data.shape) == 3:
        #     data = np.hstack([data[:,:,i].flatten().reshape((-1,1)) for i in range(data.shape[2])])
        #     print('data was reshaped to 2d')
        #     # Future: PCA
        # if reduceDim:
        #     pca = PCA(n_components=params['n_comps'])
        #     data = pca.fit_transform(data)
    
        # K = calcAffinityMat(data, params,  data_name, use_former, K_sym, graph_function, 
        #                     data = data, graph_params = graph_params,   grannet = grannet)   
        
        # if toNormRows:
        #     K = K/K.sum(1).reshape((-1,1))  
            
        # K = K - np.diag(np.diag(K) ) 

    return K
    
def mkDataGraph_grannet(data, params = {}, reduceDim = False, reduceDimParams = {}, graph_function = 'gaussian',
                K_sym  = True, use_former = False, 
                data_name = 'none', toNormRows = True,  graph_params = params_default['graph_params'],
                grannet = False):
    """
    gets 3d data and return 3d kernel
    THIS ONE IS ONLY FOR THE AVERAGE CASE OR THE WEIGHTING
    only if graph_params['kernel_grannet_type'] == "averaged" or "combination"!!!
    """                                   
                                     
    #if grannet and graph_params['kernel_grannet_type'] != 'ind' and checkEmptyList(data):
    #    raise ValueError('you must provide data for non-ind graph! but grannet type is %s'%graph_params['kernel_grannet_type'])
    non_zeros = params['n_neighbors'] + 1
    if not grannet or (grannet and graph_params['kernel_grannet_type'] == 'ind'):  # and checkEmptyList(data):
        raise ValueError('should not use this function in this case!')
    else: # in case of grannet
        """
        1) create kernel for each !!!!!!!!!!!
        """
        
        kernels_inds = np.dstack([cal_dist(k, data, graph_params = params['graph_params'], 
                                grannet = True, distance_metric = params['distance_metric'])
                        for k in range(data.shape[2])])
        
        
        #
        #np.dstack([mkDataGraph(data[:,:,trial_counter], params, reduceDim = params['reduceDim'], 
        #                     reduceDimParams = {'alg':'PCA'}, graph_function = 'gaussian',
        #            K_sym  = True, use_former = use_former, data_name = data_name,
        #            toNormRows = True,  graph_params = graph_params, grannet = grannet) for trial_counter in range(data.shape[2])])
        print('finished stage 1: calculated distances')
        """
        2)     apply average or weighting   !!!!!!!!!!!
        """
        if graph_params['kernel_grannet_type'] == "combination" :
            """
            in this case we need to calculate the shared graph
            """
            shared_kernel = cal_dist(0, MovToD2_grannet(data), graph_params = params['graph_params'], 
                                    grannet = True, distance_metric = params['distance_metric']) 
            print('finished stage 2: calculated shared ker')
            
            
            # mkDataGraph( MovToD2(data), params, reduceDim = params['reduceDim'], 
            #                      reduceDimParams = {'alg':'PCA'}, graph_function = 'gaussian',
            #             K_sym  = True, use_former = use_former, data_name = data_name,
            #             toNormRows = True,  graph_params = graph_params, grannet = grannet)
            #if graph_params['kernel_grannet_type'] != "combination" :
            dist_mat = kernel_combination(kernels_inds, shared_kernel, w = 0, graph_params = graph_params)
            print('finished stage 2: calculated combination')
            
        elif  graph_params['kernel_grannet_type'] == "averaged" :
            """
            in this case we need to calculate the shared graph
            """
            dist_mat = kernel_averaging( kernels_inds, w = [], graph_params  = params['graph_params']) 
            print('finished stage 2 calculated averaging')
            
        else:
            raise ValueError('graph_params["kernel_grannet_type"] need to be "averaged" or "combination"')
            
            
        """
        3) apply knn !!!!!!!!!!!
        """
        K = take_NN_given_dist_mat(dist_mat,non_zeros, K_sym = True, include_abs = True,  toNormRows = True)
        print('finished stage 3: took nn')
            
        """
        4) normalize kernel  !!!!!!!!!!!
        """    
        
            
        K = np.dstack([normalize_K(K[:,:,k], toNormRows = toNormRows) for k in range(K.shape[2])])
        print('finished stage 4: normalize')
        
        
        return K
    
def normalize_K(K, toNormRows = True):
    if toNormRows:
        K = K/K.sum(1).reshape((-1,1))  
        
    K = K - np.diag(np.diag(K) )
    return K
    
def calcAffinityMat(data, params,  data_name = 'none', use_former = False, K_sym = True, graph_function = 'gaussian',
                    graph_params = params_default['graph_params'], grannet = False):
    """
    THIS ONE IS ONLY 
    Calculates the affinity matrix for a given dataset using the k-nearest neighbors algorithm.

    Args:
        data (ndarray): The dataset to use for calculating the affinity matrix. data = neurons X time
        params (dict): A dictionary containing parameters for the k-nearest neighbors algorithm.
        data_name (str, optional): A string specifying the name of the dataset. Defaults to 'none'.
        use_former (bool, optional): A flag indicating whether to use the previously calculated k-nearest neighbors graph. 
            If True, the function will attempt to load the graph from disk; if False, the function will calculate a new graph. 
            Defaults to True.
        K_sym (bool, optional): A flag indicating whether to symmetrize the affinity matrix. Defaults to True.
        graph_function (str, optional): A string specifying the function to use for calculating the graph. 
            Currently, only 'gaussian' is supported. Defaults to 'gaussian'.

    Returns:
        ndarray: The affinity matrix calculated using the k-nearest neighbors algorithm.
    """    
          
    
    n_cols = data.shape[1]
    n_rows = data.shape[0]
    
    
    """
    below:
    this one is ONLY for graft (not grannet) or the grannet case of 'ind' (i.e. when params[graph_params]['kernel_grannet_type'] is ind) or one_kernel
    """
    if not grannet or len(data.shape) == 2:#graph_params['kernel_grannet_type'] in ['ind']:
            
        """
        knn_dict is a dictionary with keys 'dist' and 'ind'
        """
        knn_dict = findNeighDict(data, params, data_name, use_former, addi = '_knn', to_save = True)
    
        matInds = createSparseMatFromInd(knn_dict['ind'], is_mask = True, defVal = 1)
    
        """
        below is a sparse matrix with distances in the support
        """
        matDists = createSparseMatFromInd(knn_dict['ind'], defVal = knn_dict['dist'], is_mask = False)
    
        if graph_function == 'gaussian':
            K = gaussian_vals(matDists, std = np.median(matDists[matInds != 0 ]))
    
        else:
            raise ValueError('Unknown Graph Function')
        
        if K_sym:
            K = (K + K.T)/2
        return K
            
    else: #f graph_params['kernel_grannet_type'] in ['one_kernel']:       
        raise ValueError('how did you arrive here? if grannet and not ind type, you should use the grannet function (mkDataGraph_grannet), not this one!')
        # """
        # knn_dict is a dictionary with keys 'dist' and 'ind'
        # """
        
        # data_full_flattened = MovToD2(data_full)
        # knn_dict = findNeighDict(data_full_flattened, params, data_name, use_former, addi = '_knn', to_save = True)
    
        # matInds = createSparseMatFromInd(knn_dict['ind'], is_mask = True, defVal = 1)
    
        # """
        # below is a sparse matrix with distances in the support
        # """
        # matDists = createSparseMatFromInd(knn_dict['ind'], defVal = knn_dict['dist'], is_mask = False)
    
        # if graph_function == 'gaussian':
        #     K = gaussian_vals(matDists, std = np.median(matDists[matInds != 0 ]))
    
        # else:
        #     raise ValueError('Unknown Graph Function')            

        
        
        
        
        
        


def dist_vecs(vec1, vec2, distance_metric = 'euclidean'):
    if distance_metric == 'euclidean':
        return np.sum((vec1.flatten() - vec2.flatten())**2)
    else:
        raise ValueError('not defined yet :(')
    
    

def cal_dist_depracated(k, data_full, graph_params = params_default['graph_params'], grannet = True, distance_metric = params_default['distance_metric']):
    """
    data here is 3d: N X T X conditions
    useful for cases where graph_params['kernel_grannet_type'] is "averaged"  or "combination" 
    returns the kernel for condition k
    """
    num_rows = data_full.shape[0]
    num_conds = data_full.shape[1]
    if not grannet:
        raise ValueError('to use cal_dist you must be in grannet mode')
    if len(data_full.shape) != 3:
        input('data_full is only 2d, namely no conditions, ok?')
    
    #if distance_metric == 'euclidean': # data full is the recording for each condition: N x T x k
    if graph_params['kernel_grannet_type'] == "averaged":
        """
        below is a 3d N x N x k matrix of distances 
        """
        dists_multi_d = np.dstack([np.vstack([[dist_vecs(data_full[n,:,k], data_full[n2,:,k], distance_metric = distance_metric)
           for n2 in range(num_rows)] 
          for n in range(num_rows)] )
         for k in num_conds]) # nm future note - can make it more efficient
        
        """
        averaging
        """
        if len(graph_params['params_weights']) != num_conds  or  isinstance(graph_params['params_weights'], numbers.Number):    
            """
            weights of different kernels
            """
            if isinstance(graph_params['params_weights'], numbers.Number):
                w = [graph_params['params_weights']]*num_conds
                w[k] = 1
                w = w/np.sum(w)
            else:
                w = graph_params['params_weights']
                w = w / np.sum(w)
                
            """
            weighting the kernels
            """ 
            
            
            
        else:
            raise ValueError("graph_params['params_weights'] must be a number of a list/array with length k-1, but %s"%str(graph_params['params_weights']))
            
        graph_params['params_weights']

        
    elif graph_params['kernel_grannet_type'] == "averaged":
        oass
        
    else:
        raise ValueError('not implemented yet! (must be averaged or combination for now! but %s)'%graph_params['kernel_grannet_type'])
        
    
def cal_dist(k, data, graph_params = params_default['graph_params'], grannet = True, distance_metric = params_default['distance_metric']):
    """
    Calculates distances between time series data for GrNNEt.

    Args:
        k (int): The index of the condition to calculate distances for. WHICH CONDITION
        data (numpy.ndarray): The 3D input data for the Grannet analysis with dimensions N x T x conditions, where N is the number of nodes, T is the number of time points, and conditions is the number of conditions.
        graph_params (dict, optional): A dictionary of graph parameters to be used for the analysis. Default is `params_default['graph_params']`.
        grannet (bool, optional): Whether or not to use the Grannet method for analysis. Default is `True`.
        distance_metric (str, optional): The distance metric to be used for calculating the distances. Default is `params_default['distance_metric']`.

    Returns:
        numpy.ndarray: A 3D matrix of size N x N  containing the pairwise distances between the time series data for the specified condition. 
        
    THIS FUNCTION IS TO CALCULATE DISTANCES FOR GRANNET KERNEL GOALS
    data here is 3d: N X T X conditions
    useful for cases where graph_params['kernel_grannet_type'] is "averaged"  or "combination" 
    returns the kernel for condition k
    """
    num_rows = data.shape[0]
    num_conds = data.shape[1]
    
    if not grannet:
        raise ValueError('to use cal_dist you must be in grannet mode (this function is only for grannet)')
        
    #if len(data.shape) > 2:
    #    raise ValueError('data should not be 3d here! (cal dist)')
    """
    below is a 3d N x N matrix of distances 
    """
    if len(data.shape) == 3:
        dists_multi_d = np.vstack([[dist_vecs(data[n,:,k], data[n2,:,k], distance_metric = distance_metric)
               for n2 in range(num_rows)] 
              for n in range(num_rows)] )
    elif len(data.shape) == 2:
        print('calcultes together')
        dists_multi_d = np.vstack([[dist_vecs(data[n,:], data[n2,:], distance_metric = distance_metric)
               for n2 in range(num_rows)] 
              for n in range(num_rows)] )
        print('finished calculte together')
    return dists_multi_d
    
def kernel_averaging(data_3d, w = [], graph_params  = params_default['graph_params']) :
    
    """
    Given a 3D matrix `data_3d` of shape (N, T, k) or a list of N x T matrices, this function calculates the weighted average of the kernels for each input in the list.
    
    Parameters:
    -----------
    - data_3d : numpy ndarray or list
        The input 3D matrix of shape (N, T, k) or a list of N x T matrices.
    
    - w : list, numpy array, tuple or float, optional
        The weights used for averaging the kernels. If empty or 0, the default weights from `graph_params['params_weights']` are used. If a number, `w` is treated as the weight for all input kernels.
    
    - graph_params : dict, optional
        A dictionary containing the graph parameters. Default is `params_default['graph_params']`.
    
    Returns:
    --------
    - data_3d_weighted : numpy ndarray or list
        The weighted average of the kernels for each input in the list.
    """
    # data_3d can be a matrix of N X T X k or a list of N x T matrices
    #    this function is only for the case where graph_params['kernel_grannet_type'] == "averaged":
    # THIS FUNCTION IS CALLED AFTER!!! WE FOUND THE INDIVIDUAL KERNELS
    if graph_params['kernel_grannet_type'] != "averaged" : 
        raise ValueError('this function is only for the case where "graph_params["kernel_grannet_type"] != averaged"')


    if checkEmptyList(w) or  ( isinstance(w, numbers.Number) and  w ==0):
        w = graph_params['params_weights']  
    """
    make data a list
    """    
    if isinstance(data_3d, np.ndarray):
        data_3d_list = [data_3d[:,:,k] for k in range(data_3d.shape[2])]
        return_type = 'array'
    else:
        data_3d_list = data_3d.copy()
        return_type = 'list'
    
    """
    update w
    """
    if isinstance(w, numbers.Number):
        # if w is a number 
        w = [w]*len(data_3d)
    elif isinstance(w, (list , np.ndarray, tuple)) and len(w) == len(data_3d): # if w is a list
        pass
    else:
        raise ValueError("graph_params['params_weights'] must be a number or list with the same len as data but is %s, with len %d"%(str(graph_params['params_weights']), len(w)))
    
    
    """
    normalize w
    """   
    w_vec = w_vec / np.sum(w_vec)
    
    """
    averaging
    """    
    data_3d_weighted = [np.sum(np.dstack([data_3d_list[k_weight]*w_k[k_wight] for k_weight, w_k in enumerate(normalize_w_with_i(w_vec, k) ) ]), 2)        
        for k in range(len(data_3d_list))]   
        
    if return_type == 'array':
        data_3d_weighted = np.dstack(data_3d_wighted)  
        
    return data_3d_weighted 


def take_NN_given_dist_mat(dist_mat,non_zeros, K_sym = True, include_abs = True,  toNormRows = True):
    # given N X N matrices of the euclidean distance between neurons, calculate the 
    if isinstance(dist_mat, list) or len(dist_mat.shape) == 3:
        if isinstance(dist_mat, list):
            dist_mat = np.dstack(dist_mat)
        return np.dstack([take_NN_given_dist_mat(dist_mat[:,:,k],non_zeros) for k in range(dist_mat.shape[2])])
    
    corr_kern  = gaussian_vals(dist_mat, std = np.median(dist_mat[dist_mat != 0 ]))
    K = np.vstack([hard_thres_on_vec(corr_kern[n], non_zeros, include_abs).reshape((1,-1)) 
               for n in range(corr_kern.shape[0])])
    # if K_sym:
    #     K = (K + K.T)/2
        
    # if toNormRows:
    #     K = K/K.sum(1).reshape((-1,1))  
        
    # K = K - np.diag(np.diag(K) ) 
    
        
    return K
    

def hard_thres_on_A(A_2d, non_zeros):
    # A        should be N X T 
    """
    Apply hard thresholding on each column of the input matrix A_2d by setting 
    all entries except the non_zeros highest in absolute value to zero. Returns
    the thresholded matrix with the same shape as the input.

    Parameters:
    A_2d (ndarray): Input matrix with shape (N, T).
    non_zeros (int): Number of entries to keep after thresholding.

    Returns:
    (ndarray): Thresholded matrix with the same shape as A_2d.
    """    
    return np.hstack([hard_thres_on_vec(A_2d[:,t], non_zeros).reshape((-1,1)) for t in range(A_2d.shape[1])])
    
        
def hard_thres_on_vec(vec, non_zeros, include_abs = True)    :
    """
    Apply hard thresholding on the input vector vec by setting all entries
    except the non_zeros highest (in absolute value if include_abs=True, else
    highest in value) to zero. Returns the thresholded vector with the same 
    shape as the input.

    Parameters:
    vec (ndarray): Input vector.
    non_zeros (int): Number of entries to keep after thresholding.
    include_abs (bool): Whether to use the absolute value of entries when 
        computing the threshold. Default is True.

    Returns:
    (ndarray): Thresholded vector with the same shape as vec.
    """    
    if include_abs:
        argsort_inds = np.argsort(np.abs(vec))[::-1]
    else:
        argsort_inds = np.argsort(vec)[::-1]      

    vec[argsort_inds[non_zeros:]] = 0 # nullify small values
    return vec
       
    

def normalize_w_with_i(w,i) :
    """
    Normalize the input vector w such that its entries sum to one, then set
    the ith entry to one. Returns the normalized vector.

    Parameters:
    w (ndarray): Input vector.
    i (int): Index of the entry to set to one.

    Returns:
    (ndarray): Normalized vector with the same shape as w.
    """    
    w = w / np.sum(w)
    w[i] = 1
    w = w / np.sum(w)
    return w
    
    
    
    
    
def kernel_combination(data_3d, shared_kernel, w = 0, graph_params = params_default['graph_params']):
    """
    Combines individual and shared kernels using a weighted average.
    
    Args:
        data_3d: A numpy array of shape N x T x k or a list of N x T matrices, where N is the number of observations,
            T is the number of time steps, and k is the number of features.
        shared_kernel: A kernel of all states.
        w: The weight of the joint graph. Default is 0, which sets the weight to the value of 
            graph_params['params_weights'] if it is a number.
        graph_params: A dictionary of graph parameters. Default is params_default['graph_params'].
    
    Returns:
        A numpy array of the same shape as data_3d or a list of numpy arrays of the same shape as the 
        elements in data_3d, representing the combined kernels.
        
    THIS FUNCTION IS CALLED AFTER!!! WE FOUND THE INDIVIDUAL KERNELS
    """
    # data_3d can be a matrix of N X T X k or a list of N x T matrices
    # this function takes a combination if individual and shared kernel
    # w is the weight of the joint graph
    
    # shared kernel = kerenel of all states
    # w decides the weight unless 0
    if graph_params['kernel_grannet_type'] != "combination" :
        raise ValueError('this function is only for the case where "graph_params["kernel_grannet_type"] != combination"')
    if w ==0:
        if isinstance(graph_params['params_weights'], numbers.Number):
            w = graph_params['params_weights']
            
        else:
            raise ValueError("graph_params['params_weights'] must be a number but is %s"%str(graph_params['params_weights']))
    
    w_vec = np.array([1, w])
    w_vec = w_vec / np.sum(w_vec)
    if isinstance(data_3d, np.ndarray):
        data_3d_list = [data_3d[:,:,k] for k in range(data_3d.shape[2])]
        return_type = 'array'
    else:
        data_3d_list = data_3d.copy()
        return_type = 'list'
    
    data_3d_weighted =  [w_vec[0]* data_3d_spec + w_vec[1]*shared_kernel for data_3d_spec in data_3d_list]
    if return_type == 'array':
        data_3d_weighted = np.dstack(data_3d_weighted)
    
        
    return data_3d_weighted
    
    
    
    
    
def findNeighDict(data, params, data_name = 'none', 
                    use_former = False, addi = '_knn', to_save = True):
    """
    this one is ONLY for graft (not grannet) or the grannet case of 'ind' (i.e. when params[graph_params]['kernel_grannet_type'] is ind)
    """
    save_knn_path = data_name + '%s.npy'%addi #np.save()
    if use_former and os.path.isfile(save_knn_path) :
        print('used former')
        knn_dict = np.load(save_knn_path, allow_pickle=True).item()
    else:
        if params['n_neighbors'] > data.shape[1]:
            print('Too many neighbors were required, set it to %d'%int(data.shape[1]/2))
            params['n_neighbors'] = int(data.shape[1]/2)
        if params['n_neighbors'] < 1:
            params['n_neighbors'] = int(params['n_neighbors']*data.shape[1])
        """
        return a matrix of samples X neighs
        """
        nbrs = NearestNeighbors(n_neighbors=  params['n_neighbors'] + 1, 
                                algorithm=params['alg']).fit(data)
        distances, indices = nbrs.kneighbors(data)
        """
        explanation:
            distances - num samples  x num neighbors array
            indices   - num samples  x num neighbors array of indixes, 1st col is the same el
        """
        knn_dict = {'dist': distances, 'ind': indices}
        # print('indices size')
        # print(str(indices.shape))
        if to_save:
            np.save(save_knn_path, knn_dict)
    return knn_dict
    
    
def createSparseMatFromInd(inds, M = 0, defVal = 1, is_mask = True ):

    """
    This function find a 0-1 matrix where the non-zeros are located according to inds
    Parameters
    ----------
    inds : np.ndarray [sample index X number of neighbors]
        indices of the neighbors
    M : int, optional
        DESCRIPTION. The default is 0.
    defVal : number OR numpy.ndarray with the same shape of inds, optional
        DESCRIPTION. The default is 1.        

    Returns
    -------
    mat :  np.ndarray of size M X M of 0/1 values

    """
    if M == 0 or M < np.max(inds):
        M = np.max([np.max(inds)+1, inds.shape[0]])
        print('M was changed in "createSparseMatFromInd"')
    mat = np.zeros((M,M))
    # print('M')
    # print(M)
    if not is_mask: mat += np.inf # (check_effect)
    #print(mat)
    rows = np.arange(inds.shape[0]) #np.repeat(np.arange(inds.shape[0]).reshape((-1,1)),inds.shape[1], axis=1)   
    for row in rows:
        col_column = inds[row,:]
        #print(col_column)
        for col_count, col in enumerate(col_column):
            #print(col)
            if isinstance(defVal, np.ndarray):
                # print([row,col])
                # print(defVal[row,col_count])
                mat[row,col] = defVal[row,col_count]
            else:
                mat[row,col] = defVal    
            # print(mat)
            # input('?')
    return mat


    

def gaussian_vals(mat, std = 1, mean = 0 , norm = False, dimensions = 1, mat2 = [], power = 2):
    """
    check_again
    Parameters
    ----------
    mat : the matrix to consider
    std : number, gaussian std
    mean : number, optionalis 
        mean gaussian value. The default is 0.
    norm : boolean, optional
        whether to divide values by sum (s.t. sum -> 1). The default is False.

    Returns
    -------
    g : gaussian values of mat

    """    
    if dimensions == 1:
        if not checkEmptyList(mat2): warnings.warn('Pay attention that the calculated Gaussian is 1D. Please change the input "dimensions" in "gaussian_vals" to 2 if you want to consider the 2nd mat as well')
        # print('std')
        # print(mean)
        # print('mean')
        # print(std)
        # input('mean and std')
        g = np.exp(-((mat-mean)/std)**power)
        if norm: return g/np.sum(g)

    elif dimensions == 2:
        #dim1_mat = np.abs(mat1.reshape((-1,1)) @ np.ones((1,len(mat1.flatten()))))
        #dim2_mat = np.abs((mat2.reshape((-1,1)) @ np.ones((1,len(mat2.flatten())))).T)
        #g= np.exp(-0.5 * (1/std)* (dim1_mat**power + (dim1_mat.T)**power))
        g = gaussian_vals(mat, std , mean , norm , dimensions = 1, mat2 = [], power = power)
        g1= g.reshape((1,-1))
        g2 = np.exp(-0.5/np.max([int(len((mat2-1)/2)),1])) * mat2.reshape((-1,1))
        g = g2 @ g1 
        
        g[int(g.shape[0]/2), int(g.shape[1]/2)] = 0
        if norm:
            g = g/np.sum(g)
        
    else:
        raise ValueError('Invalid "dimensions" input')
    return g
        
    
#%% GD Updates
def update_GDiters(dict_old, A, data, step_s, params):
    """
    Take a step in the negative gradient of the basis: Minimizing the energy E = ||x-Da||_2^2 + lambda*||a||_1^2

    Parameters
    ----------
    dict_old : TYPE
    A : TYPE
    data : TYPE
    step_s : TYPE
        DESCRIPTION.
    l2 : TYPE
        DESCRIPTION.
    params : TYPE,
        DESCRIPTION.

    Returns - new dict
    -------
    """
    for index2 in range(params.get('GD_iters')):
        # Update The basis matrix

        dict_old = dict_old + (step_s/A.shape[0])*((data.T - dict_old @ A.T) @ A) 

    # This part is basically the same, only for the hyyperspectral, care needs to be taken to saturate at 0,
    # so that no negative relflectances are learned. 
    if params.get('nonneg'):
        dict_old[dict_old < 0] = 0 + epsilon
        if np.sum(dict_old) ==0:
            raise ValueError('fgdgdfgdfgfgfdgfgfdglkfdjglkdjglkfdjglkdjflgk')
    return dict_old

def update_FullLS_Poisson(dict_old, A, data,  params):
    raise ValieError('Undefind least squares for Poisson - full_ls')

def update_LSwithContForb_Poisson(dict_old, A, data, l2, l3, params, step_s):  
    """
    E = ||X-DA||_2^2 + l3*||D_old - D||_F^2 + l2*||D||_F^2 
    Take a step in the negative gradient of the basis: Minimizing the energy E = ||x-Da||_2^2 + lambda*||a||_1^2

    Parameters
    ----------
    dict_old : TYPE
    A : TYPE
    data : TYPE
    step_s : TYPE
        DESCRIPTION.
    l2 : TYPE
        DESCRIPTION.
    params : TYPE,
        DESCRIPTION.

    Returns - new dict
    -------
    """
    dict_old_exist = 0
    dict_old_prev = np.zeros(dict_old.shape)
    for index2 in range(params.get('GD_iters')):
        # Update The basis matrix
        # % Take a step in the negative gradient of the basis:
        # % Minimizing the energy:
        # % lambda = phi*A.T 

        # % Update The basis matrix
        denomi = (dict_old @ A.T).T;
        grad_matrix = gradient_log(A, data, denomi);       
        # %fprintf('lambda %s /n A %s /n X_im %s /n dict old %s /n', size(lambda), size(A), size(data), size(dict_old));
        # %dict_new = dict_old - (step_s/size(A,1))*...
        # %    (lambda - data.*log(lambda))*(A - pinv(lambda)*data*A );
                    #dict_old = dict_old - (step_s/A.shape[0]) @ (-grad_matrix + np.ones((data.shape[1],1)) @ np.ones((1, A.shape[0])) @ A)
        non_reg_term = (step_s/A.shape[0])*(-grad_matrix + np.ones((x_im.shape[0],1)) @ np.ones((1, A.shape[0])) @ A)
        dict_new = dict_old - non_reg_term - (step_s/A.shape[0])*l3*dict_old_exist @ (dict_old - dict_old_prev) - step_s*l2/coef_vals.shape[0]*dict_old
        dict_old_prev = dict_old
        dict_old = dict_new
        dict_old_exist = 1
        # This part is basically the same, only for the hyyperspectral, care needs to be taken to saturate at 0,
        # so that no negative relflectances are learned. 
        if params.get('nonneg'):
            dict_old[dict_old < 0] = 0 + epsilon
            if np.sum(dict_old) == 0:
                raise ValueError('All data values are 0')
    return dict_old


def full_ls_cor_poisson(dict_old, A, data, l2, l3, l4,params, step_s):  
    """
    E = ||X-DA||_2^2 + l3*||D_old - D||_F^2 + l2*||D||_F^2 
    Take a step in the negative gradient of the basis: Minimizing the energy E = ||x-Da||_2^2 + lambda*||a||_1^2
    (in matlab located in  dictionary_update()

    Parameters
    ----------
    dict_old : TYPE
    A : TYPE
    data : TYPE
    step_s : TYPE
        DESCRIPTION.
    l2 : TYPE
        DESCRIPTION.
    params : TYPE,
        DESCRIPTION.

    Returns - new dict
    -------
    """
    dict_old_exist = 0
    dict_old_prev = np.zeros(dict_old.shape)
    for index2 in range(params.get('GD_iters')):
        # Update The basis matrix
        # % Take a step in the negative gradient of the basis:
        # % Minimizing the energy:
        # % lambda = phi*A.T
         

        # % Update The basis matrix
        denomi = (dict_old @ A.T).T
        grad_matrix = gradient_log(A, data, denomi);       
        # %fprintf('lambda %s /n A %s /n X_im %s /n dict old %s /n', size(lambda), size(A), size(data), size(dict_old));
        # %dict_new = dict_old - (step_s/size(A,1))*...
        # %    (lambda - data.*log(lambda))*(A - pinv(lambda)*data*A );
                    #dict_old = dict_old - (step_s/A.shape[0]) @ (-grad_matrix + np.ones((data.shape[1],1)) @ np.ones((1, A.shape[0])) @ A)
        non_reg_term = (step_s/A.shape[0])*(-grad_matrix + np.ones((data.shape[1],1)) @ np.ones((1, A.shape[0])) @ A)
        
        # print('reg_term')
        # print(non_reg_term.shape)
        # print('dict_old')
        # print(dict_old.shape)
        # print('dict_old_prev')
        # print(dict_old_prev.shape)
        # #print(((step_s/A.shape[0])*l3*dict_old_exist).shape)
        # print( (dict_old - dict_old_prev).shape)

        l3_reg = (step_s/A.shape[0])*l3*dict_old_exist * (dict_old - dict_old_prev)
        l2_reg = step_s*l2/A.shape[0]*dict_old  # A is coef_vals? 
        # print(type(l3_reg))
        # print(type(l2_reg))
        # print(type(step_s))
                
        dict_new = dict_old - non_reg_term - l3_reg - l2_reg - (step_s/A.shape[0])*l4*(dict_old @ np.ones((dict_old.shape[1], dict_old.shape[1])) -  dict_old)   ## A is coef_vals? 
        dict_old_prev = dict_old
        dict_old = dict_new
        dict_old_exist = 1
        # This part is basically the same, only for the hyyperspectral, care needs to be taken to saturate at 0,
        # so that no negative relflectances are learned. 
        if params.get('nonneg'):
            dict_old[dict_old < 0] = 0 + epsilon
            if np.sum(dict_old) == 0:
                raise ValueError('All data values are 0')
    return dict_old



    

def update_LSwithCont_Poisson(dict_old, A, data, step_s, l3, params):
   
    """

    E = ||lambda - ylog(lambda)||_2^2 + l3*||D - D_old||_F^2
    Take a step in the negative gradient of the basis: Minimizing the energy E = ||x-Da||_2^2 + lambda*||a||_1^2

    Parameters
    ----------
    dict_old : TYPE
    A : TYPE
    data : TYPE
    step_s : TYPE
        DESCRIPTION.
    l2 : TYPE
        DESCRIPTION.
    params : TYPE,
        DESCRIPTION.

    Returns - new dict
    -------
    """
    dict_old_exist = 0
    dict_old_prev = np.zeros(dict_old.shape)
    for index2 in range(params.get('GD_iters')):
        # Update The basis matrix
        # % Take a step in the negative gradient of the basis:
        # % Minimizing the energy:
        # % lambda = phi*A.T
         

        # % Update The basis matrix
        denomi = (dict_old @ A.T).T
        grad_matrix = gradient_log(A, data, denomi);       
        # %fprintf('lambda %s /n A %s /n X_im %s /n dict old %s /n', size(lambda), size(A), size(data), size(dict_old));
        # %dict_new = dict_old - (step_s/size(A,1))*...
        # %    (lambda - data.*log(lambda))*(A - pinv(lambda)*data*A );
                    #dict_old = dict_old - (step_s/A.shape[0]) @ (-grad_matrix + np.ones((data.shape[1],1)) @ np.ones((1, A.shape[0])) @ A)
        non_reg_term = (step_s/A.shape[0])*(-grad_matrix + np.ones((x_im.shape[0],1)) @ np.ones((1, A.shape[0])) @ A)
        dict_new = dict_old - non_reg_term - (step_s/A.shape[0])*l3*dict_old_exist @ (dict_old - dict_old_prev)
        dict_old_prev = dict_old
        dict_old = dict_new
        dict_old_exist = 1
        # This part is basically the same, only for the hyyperspectral, care needs to be taken to saturate at 0,
        # so that no negative relflectances are learned. 
        if params.get('nonneg'):
            dict_old[dict_old < 0] = 0 + epsilon
            if np.sum(dict_old) == 0:
                raise ValueError('All data values are 0')
    return dict_old


def update_GDiters_Poisson(dict_old, A, data, step_s, params):
    """
    Take a step in the negative gradient of the basis: Minimizing the energy E = ||x-Da||_2^2 + lambda*||a||_1^2

    Parameters
    ----------
    dict_old : TYPE
    A : TYPE
    data : TYPE
    step_s : TYPE
        DESCRIPTION.
    l2 : TYPE
        DESCRIPTION.
    params : TYPE,
        DESCRIPTION.

    Returns - new dict
    -------
    """
    for index2 in range(params.get('GD_iters')):
        # Update The basis matrix
        # % Take a step in the negative gradient of the basis:
        # % Minimizing the energy:
        # % lambda = phi*A.T
        # % E = ||lambda - ylog(lambda||_2^2 

        # % Update The basis matrix
        denomi = (dict_old @ A.T).T;
        grad_matrix = gradient_log(A, data, denomi);       
        # %fprintf('lambda %s /n A %s /n X_im %s /n dict old %s /n', size(lambda), size(A), size(data), size(dict_old));
        # %dict_new = dict_old - (step_s/size(A,1))*...
        # %    (lambda - data.*log(lambda))*(A - pinv(lambda)*data*A );

        #dict_old - (step_s/A.shape[0])
        dict_old = dict_old - (step_s/A.shape[0]) * (-grad_matrix + 
                                                     np.ones((data.shape[1],1)) @ np.ones((1, A.shape[0])) @ A)

    # This part is basically the same, only for the hyyperspectral, care needs to be taken to saturate at 0,
    # so that no negative relflectances are learned. 
    if params.get('nonneg'):
        dict_old[dict_old < 0] = 0 + epsilon
        if np.sum(dict_old) == 0:
            #raise ValueError('All data values are 0')
            warnings.warn('All data values are 0')
            dict_old = dict_old + params['sigma_noise']*np.random.randn(*dict_old.shape)
    return dict_old

# def gradient_log(A, data, denomi):
#     """
#     Calculate the gradient of sum_ij(y_ij.*log([phi*A.T]_ij)
    
#     Parameters:
#     -----------
#     A : numpy.ndarray
#         The mixing matrix with shape (n, p), where n is the number of pixels and p is the number of components.
#     data : numpy.ndarray
#         The data matrix with shape (n, T), where T is the number of time points.
#     denomi : float
#         The denominator used to calculate the c_matrix (page 13 in the original work).
        
#     Returns:
#     --------
#     mat_gradient : numpy.ndarray
#         The gradient matrix with shape (T, p).
#     """    
#     # this is function \begin{multline}
#     # \frac{\partial g_2(y,\Phi,A)}{\partial \Phi} &= \frac{\partial \left(\sum\limits_{ti}-y_{ti}\log\left(\Phi_{A^T}{ti}\right)\right)}{\partial \Phi} \
#     # &= -\sum\limits_{ti}\frac{y_{ti}}{[\Phi {A^T}]_{ti}}\delta_{(t,i),(T,N)}A \
#     # &= -\sum\limits_{ti}c_{ti}\delta_{(t,i),(T,N)}A \
#     # &= -\sum\limits_{i}\delta_{(i,T)}\left[1\right]_{(1,N)}\text{diag}\left(\left[C\right]_i\right)A
#     # \end{multline}
    
#     # % This function calculate the gradient of sum_ij(y_ij.*log([phi*A.T]_ij) p
#     # % mat_gradient: [m X p]
#     # data: [n X T]
#     # % data.T: [T X n]
#     # % lambda = phi * A.T : [m X n]
#     # % A: [n X p]
#     # % p = number of components; T = number of time points; n = number of pixels

#     
#     #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#     n = A.shape[0]
#     p = A.shape[1]
#     T = data.shape[1]
    
#     c_matrix = data/denomi; # page 13 in my work, c_matrix is [N X T]
    
#     # mat_gradient = np.zeros((T,p));
#     # for i in range(T): # for every time point
#     #    # diag_mat = np.diag(c_matrix[:,i]);
#     #    weighted_sum_rows_A = np.sum(diag_mat @ A, 0)
#     #    mat_gradient[i,:] = weighted_sum_rows_A
#     #    mat_gradient[np.isnan(mat_gradient)] = 0

#     return c_matrix @ A

def gradient_log(A, data, denomi):  
    """
    Compute the gradient of a log-likelihood function with respect to the parameter matrix A.

    Parameters:
    A (numpy.ndarray): A parameter matrix of size [N x P].
    data (numpy.ndarray): A data matrix of size [T x N].
    denomi (numpy.ndarray): A vector of size [T x N] that serves as the denominator in the log-likelihood function.

    Returns:
    gradient (numpy.ndarray): The gradient of the log-likelihood function with respect to A, which is a matrix of size [N x T].

    """    
   
    
    c_matrix = data/denomi; # page 13 in my work, c_matrix is [N X T]
    
    if c_matrix.shape[1] == A.shape[0]: 
        return c_matrix @ A
    else:
        #raise ValueError('shape problems! c_mat shape is %s but A shape is %s'%(c_matrix.shape, A.shape))
        c_matrix = c_matrix.T
        print('transpose c')
        return c_matrix @ A



def update_GDwithForb(dict_old, A, data, step_s, l2, params):
    """
    Take a step in the negative gradient of the basis:
    This time the Forbenious norm is used to reduce unused basis elements. The energy function being minimized is then:
    E = ||x-Da||_2^2 + lambda*||a||_1^2 + lamForb||D||_F^2
    
    Parameters
    ----------
    dict_old : TYPE
    A : TYPE
    data : TYPE
    step_s : TYPE
        DESCRIPTION.
    l2 : TYPE
        DESCRIPTION.
    params : TYPE
        DESCRIPTION.

    Returns - new dict
    -------
    """
    for index2 in range(params.get('GD_iters')):
        # Update The basis matrix
        dict_new = dict_old + (step_s)*((data.T - dict_old @ A.T) @ A -l2*dict_old) @ np.diag(1/(1+np.sum(A != 0, 0)));
    
        # For some data sets, the basis needs to be non-neg as well
        if params.get('nonneg'):
            dict_new[dict_new < 0] = 0 + epsilon
    return dict_new
    
def update_FullLS(dict_old, A, data, params):
    """
    Minimizing the energy:
    E = ||X-DA||_2^2 via D = X*pinv(A)
    
    Parameters
    ----------
    dict_old : TYPE
    A : TYPE
    data : TYPE
    params : TYPE

    Returns
    -------     dict_new
    """
    raise ValueError('how did you arrive here?')
    if params.get('nonneg'):
        dict_new = np.zeros(size(dict_old))                                  # Initialize the dictionary
        n_times = dict_old.shape[0]
        for  t in range(n_times):
            dict_new[t,:] = nnls(A, data[:,t]) # Solve the least-squares via a nonnegative program on a per-dictionary level                   
    else:
        dict_new = data.T @ np.pinv(A);                                         # Solve the least-squares via an inverse

    return  dict_new


def  update_LSwithCont(dict_old, A, data, l3, params):    
    """    
    Minimizing the energy:    E = ||X-DA||_2^2 + l3*||D_old - D||_F^2 via D = [X;D_old]*pinv([A;I])    
    
    Parameters
    ----------
    dict_old : TYPE
    A : TYPE
    data : TYPE
    l2 : TYPE
    l3 : TYPE
    params : TYPE

    Returns
    -------     dict_new
    """
    if params.get('nonneg'):
        dict_new = np.zeros(dict_old.shape)                                      # Initialize the dictionary
        n_times = dict_old.shape[0]
        n_neurons = A.shape[0]
        for t  in range(n_times):
            dict_new[t,:] = nnls( np.vstack([A.T, l3*np.eye(n_neurons)  ]),
                                            np.vstack([data[:,t].reshape((-1,1)),
                                                       l3*dict_old[t,:].reshape((-1,1)) ]) );     # Solve the least-squares via a nonnegative program on a per-dictionary level

    else:
        dict_new = np.vstack([data,l3*dict_old.T,l2*dict_old]) @ np.linalg.pinv(np.vstack([A.T,l3*np.eye(n_neurons)])) # Solve the least-squares via an inverse
    return  dict_new
 
def update_LSwithContForb(dict_old, A, data, l2, l3, params):
    """
    Minimizing the energy:
    E = ||data.T-DA.T||_2^2 + l3*||D_old - D||_F^2 + l2*||D||_F^2 ,                      via phi = [data.T;phi_old]*pinv([A.T;I])
    
    Parameters
    ----------
    dict_old : TYPE
    A : TYPE
    data : TYPE
    l2 : TYPE
    l3 : TYPE
    params : TYPE

    Returns
    -------
    dict_new
    """
    if params.get('nonneg'):
        dict_new = np.zeros(dict_old.shape)                                      # Initialize the dictionary
        n_times = dict_old.shape[0]
        n_neurons = A.shape[0]
        for t  in range(n_times):
            dict_new[t,:] = nnls( np.vstack([A.T, l3*np.eye(n_neurons),np.zeros((n_neurons, n_neurons))  ]),
                                            np.vstack([data[:,t].reshape((-1,1)),
                                                       l3*dict_old[t,:].reshape((-1,1)),
                                                      l2*dict_old[t,:].reshape((-1,1)) ]) );     # Solve the least-squares via a nonnegative program on a per-dictionary level

    else:
        dict_new = np.vstack([data,l3*dict_old.T,l2*dict_old]) @  np.linalg.pinv(np.vstack([A.T,l3*np.eye(n_neurons),zeros((n_neurons, n_neurons))])); # Solve the least-squares via an inverse
    return  dict_new
     
def   update_LSwithForb(dict_old, A, data, l2, params):
    """
    Minimizing the energy:
    E = ||X-DA||_2^2 + l2*||D||_F^2
                 via  D = X*A^T*pinv(AA^T+lamForb*I)
    
    Parameters
    ----------
    dict_old : np.ndarray, T X p
        temporal profiles dict
    A : np.ndarray, N X p
        neural nets
    data : np.ndarray, N X T
        neural recordings
    l2: number (regularization)
    params : options

    Returns
    -------
    dict_new : np.ndarray, T X p
        temporal profiles dict

    """

    if params.get('nonneg'):
        #future
        warnings.warn('Regularized non-negative ls is not implemented yet! Solving without non-negative constraints.../n')  
    dict_new = data.T @ A @ np.linalg.pinv(A.T @ A + l2*np.eye(A.shape[1])); # Solve the least-squares via an inverse
    return dict_new
    
def  update_FullLsCor(dict_old, A, data, l2, l3, l4, params):
    """
    E = ||X-DA||_2^2 + l4*||D.'D-diag(D.'D)||_sav + l2*||D||_F^2 + l3*||D-Dold||_F^2 
        
    Parameters
    ----------
    dict_old : np.ndarray, T X p
        temporal profiles dict
    A : np.ndarray, N X p
        neural nets
    data : np.ndarray, N X T
        neural recordings
    l2, L3, L4 : numbers (regularization)
    params : options

    Returns
    -------
    dict_new : np.ndarray, T X p
        temporal profiles dict

    """
    if params.get('nonneg'): # if non-negative matrix factorization
        dict_new = np.zeros(dict_old.shape);                                  # Initialize the dictionary
        n_nets    = dict_old.shape[1]
        n_times    = dict_old.shape[0]   

        # Solve the least-squares via a nonnegative program on a per-dictionary level
        for t in range(n_times): #efficiencyMarker
            dict_new[t,:] = solve_qp(2*A.T @ A + l4 + (l3+l2-l4)*np.eye(n_nets), 
                                     ( -2*A.T @ data[:,t] + l3*dict_old[t,:]).reshape((1,-1)) 
                                     , solver = params['solver_qp'] )

    else:
        if np.max(A) < 10**8:
            dict_new = data.T @ A @ np.linalg.pinv(A.T @ A + l4*(1-np.eye(A.shape[1]))) ; # Solve the least-squares via an inverse
        else:
            A = A/np.median(A)
            if np.max(A) > 10**8:
                A = A/np.max(A)
            A = A+np.random.randn(*A.shape)*0.05*np.mean(np.abs(A))
            dict_new = data.T @ A @ np.linalg.pinv(A.T @ A + l4*(1-np.eye(A.shape[1]))) ; 
    return dict_new

def sparseDeconvDictEst(dict_old, A, data, l2, params):
    """
    This function should return the solution to the optimiation
    S = argmin[||A - (l2*S)data||_F^2 + ]
    D_i = l2*S_i
    
    
    Parameters
    ----------
    dict_old : np.ndarray, T X p
        temporal profiles dict
    A : np.ndarray, N X p
        neural nets
    data : np.ndarray, N X T
        neural recordings
    l2 : number (regularization)
    params : dict
    
    Returns
    -------
    phi : np.ndarray, T X p
        temporal profiles dict

    """
    

    #return phi
    raise ValueError('Function currently not available. Please change the "GD_type" from "sparse_deconv"')
    pass




    
    
    
    
    
    
#%%  Other pre-processing
def norm_mat(mat, type_norm = 'evals', to_norm = True):
  """
  This function comes to norm matrices by the highest eigen-value
  Inputs:
      mat       = the matrix to norm
      type_norm = what type of normalization to apply. Can be 'evals', 'unit' or 'none'.
      to_norm   = whether to norm or not to.
  Output:  
      the normalized matrix
  """    
  if to_norm and type_norm != 'none':
    if type_norm == 'evals':
      eigenvalues, _ =  linalg.eig(mat)
      mat = mat / np.max(np.abs(eigenvalues))
    elif type_norm == 'unit':
      mat = mat @ np.diag(1 / np.sqrt(np.sum(mat**2,0))) 
         
  return mat


#%% Plotting Functions

def visualize_images(to_use_array = True, to_norm = True,
        folder_path =  r'E:/CODES FROM GITHUB/GraFT-analysis/code/neurofinder.02.00/images'  ):
    """
    Visualize a stack of images from a folder by displaying a slider to select
    a single image. The folder should contain image files in tiff format. If 
    to_use_array is True, the images will be loaded into a numpy array before 
    display. If to_norm is True, the array will be normalized by the maximum 
    pixel value along each dimension.

    Parameters:
    to_use_array (bool): Whether to load the images into a numpy array. 
        Default is True.
    to_norm (bool): Whether to normalize the array by the maximum pixel value 
        along each dimension. Default is True.
    folder_path (str): Path to the folder containing the images. Default is 
        r'E:/CODES FROM GITHUB/GraFT-analysis/code/neurofinder.02.00/images'.
    """    
    if to_use_array: 
        array_images =  from_folder_to_array(folder_path)
        if to_norm:
            array_images = array_images/np.maximum([np.max(array_images,0 ), np.max(array_images,1)]).reshape((1,1,-1))
    return array_images
            
    
    
def from_folder_to_array(path_images =  r'E:/CODES FROM GITHUB/GraFT-analysis/code/neurofinder.02.00/images' 
                         , max_images = 100):
    """
    Load a stack of images from a folder into a numpy array. The folder should 
    contain image files in tiff format.

    Parameters:
    path_images (str): Path to the folder containing the images. Default is 
        r'E:/CODES FROM GITHUB/GraFT-analysis/code/neurofinder.02.00/images'.
    max_images (int): Maximum number of images to load. Default is 100.

    Returns:
    (ndarray): 3D array of shape (H, W, T), where H is the image height, W is
        the image width, and T is the number of images loaded.
    """    
    if isinstance(path_images,(np.ndarray, list)):
        pass
    elif isinstance(path_images,str):
        files = os.listdir(path_images)
        files = np.sort(files)
    return np.dstack([load_image_to_array(path_images = path_images, image_to_load = cur_file) for counter, cur_file in enumerate(files) if counter < max_images])
    
    
def load_image_to_array(path_images =  r'E:/CODES FROM GITHUB/GraFT-analysis/code/neurofinder.02.00/images',
               image_to_load = 'image07971.tiff'):
    
    """
    Load a single image from a folder into a numpy array. The image should be 
    in tiff format.

    Parameters:
    path_images (str): Path to the folder containing the image. Default is 
        r'E:/CODES FROM GITHUB/GraFT-analysis/code/neurofinder.02.00/images'.
    image_to_load (str): Filename of the image to load. Default is 
        'image07971.tiff'.

    Returns:
    (ndarray): 2D array of shape (H, W), where H is the image height and W is
        the image width.
    """
    
    im_path = path_images + '/%s'%image_to_load
    im = io.imread(im_path)
    imarray = np.array(im)
    return imarray

    
def slider_changed(event):  
    """
    Callback function for a slider that changes the displayed image in a 
    matplotlib Axes object.

    Parameters:
    event: Event object passed by the slider widget.
    """    
    
    val = slider.get()
    ax.imshow(array_images[:,:,int(val)])

#%% Working with files
    
def load_mat_file(mat_name , mat_path = '',sep = sep, squeeze_me = True,simplify_cells = True):
    """
    Function to load mat files. Useful for uploading the c. elegans data. 
    Example:
        load_mat_file('WT_Stim.mat','E:/CoDyS-Python-rep-/other_models')
    """
    if mat_path == '':
        data_dict = sio.loadmat(mat_name, squeeze_me = squeeze_me,simplify_cells = simplify_cells)
    else:
        data_dict = sio.loadmat(mat_path+sep+mat_name, squeeze_me = True,simplify_cells = simplify_cells)
    return data_dict    
    
#%% Data Pre-Processing    
 
def take_df_under_col(df, col_name, val_col):
    """
    Returns a subset of a pandas DataFrame where the value in the specified column matches the given value.

    Parameters:
    df (pandas.DataFrame): the DataFrame to filter
    col_name (str): the name of the column to match against
    val_col (any): the value to match in the column

    Returns:
    pandas.DataFrame: a subset of the original DataFrame where the specified column has the given value
    """    
    return df.iloc[np.where(df[col_name] ==  val_col)[0],:]

def to_smooth_data(data, kernel = 'gaussian', direction = 0, 
                   kernel_size_take = 40,
                   kernel_size = 40, freq = 150,
                kernel_params = {'std' : 1, 'mean' : 0}):
    """
    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    kernel : can be a string or a np.array or a list, optional
        if string, currently, can be ony 'gaussian'. The default is 'gaussian'.
    direction: if data is 2d

    Returns
    -------
    smoothed_data : TYPE
        DESCRIPTION. 
    """
    if direction not in [0,1]: raise ValueError('Invalid Direction!') 
        
    if isinstance(kernel , str):            
        if kernel == 'gaussian':
            if np.mod(kernel_size,2) == 0: kernel_size =kernel_size + 1
            mat = np.arange(-(kernel_size-1)/2, (kernel_size-1)/2)/freq
            kernel = gaussian_vals(mat, norm = True, power = 2, **kernel_params)
            take_in_kernel = np.round(np.linspace(0,kernel_size-2, kernel_size_take + 2)).astype(int)
            kernel = kernel[take_in_kernel]
            
        else:
            raise ValueError('Unknown kernel type!')
    if len(data.shape) == 3: # Here the 3rd dim must be the repeatitions
        smoothed_data = np.dstack([to_smooth_data(data[:,:,i], kernel, direction, kernel_size, freq, 
                    kernel_params) for i in range(data.shape[2])])
        
    elif len(data.shape) == 2:
        if len(kernel) >= data.shape[direction]:
            raise IndexError('Kernel should be shorter than the data')
        if direction == 0:
            smoothed_data = np.hstack([np.convolve(data[:,col], kernel, mode = 'same').reshape((-1,1))
                                       for col in np.arange(data.shape[1]) ]  )
        elif direction == 1 :
            smoothed_data = np.vstack([np.convolve(data[row,:], kernel, mode = 'same').reshape((1,-1))
                                       for row in np.arange(data.shape[0]) ]  )

            
    elif len(data.shape) == 1:
        if len(kernel) >=  len(data):
            raise IndexError('Kernel should be shorter than the data')
        smoothed_data = np.convolve(data, kernel, mode = 'same')      
    else:
        raise ValueError('Invalid Data Dimension for Smoothing')
    
    return smoothed_data
    
    
    

def from_counted_dict_to_array(counted_dict,  to_make_array = True):
    """
    Converts a dictionary with values that are pandas DataFrames of the same shape into a 3D numpy array.
    
    Args:
        counted_dict (dict): Dictionary with keys that are integers and values that are pandas DataFrames of the same shape.
        to_make_array (bool): If True, converts the dictionary into a 3D numpy array. If False, returns the original dictionary.
    
    Returns:
        If to_make_array is True, returns a 3D numpy array with the values of the input dictionary stacked along the third axis.
        If to_make_array is False, returns the input dictionary.
    """
    if to_make_array:
        keys_sorted = np.sort(list(counted_dict.keys()))
        #array_stack = np.empty((counted_dict[key_sorted[0]].shape[0],counted_dict[key_sorted[0]].shape[1], keys_sorted))
        return np.dstack([counted_dict[key].values.reshape((counted_dict[key].shape[0],counted_dict[key].shape[1], 1))
                                   for key in keys_sorted])
    return counted_dict
        

    
def plot_grphs_networks(type_mapping = 'trends'):
    """
    Plots graphs for each column of a matrix, where each column represents a directed graph. 
    The edges of the graphs are represented by the non-zero entries in the corresponding column, and 
    the nodes are labeled using a mapping provided as input.
    
    Args:
        type_mapping (str): Determines the mapping to be used. If 'trends', the function will use the mapping stored
                            in the file 'mapping_trends.npy' in the current directory.
    
    Returns:
        None
    """
    if type_mapping == 'trends':
        mapping = np.load('mapping_trends.npy', allow_pickle=True).item()
        path_result = r'E:\CODES FROM GITHUB\GraFT_Python\GraFT_Python\trends\2022-10-31\kk.npy'
        # new trends is in 
        
    n = np.load(path_result, allow_pickle=True).item()
    na = n['A']/n['A'].sum(0)
    fig, axs = plt.subplots(1, na.shape[1], figsize = (20,5))
    for col_num in np.arange(na.shape[1]):
        col = na[:, col_num]
        graph_one_col(col, mapping, axs[col_num])
        #G = nx.relabel_nodes(G, mapping, copy = False)
        
    
    #na_sym = na[:,0].reshape((-1,1,)) @ na[:,0].reshape((1,-1))

def graph_one_col(col_vals, mapping, ax, perc_null = 99):
    """
    Plots a graph corresponding to the non-zero entries in a given column of a matrix.

    Args:
        col_vals (ndarray): 1D numpy array representing the column to be plotted.
        mapping (dict): A dictionary mapping the indices of the nodes in the graph to their labels.
        ax (AxesSubplot): The matplotlib axes on which to plot the graph.
        perc_null (float): Percentage of edges with the smallest weights to be discarded.

    Returns:
        None
    """
    na_sym = col_vals.reshape((-1,1,)) @ col_vals.reshape((1,-1))    
    na_sym[na_sym < np.percentile(na_sym,perc_null)] =0
    na_sym = na_sym - np.diag(np.diag(na_sym))
    G = nx.DiGraph(na_sym)
    G = nx.relabel_nodes(G, mapping, copy = False)
    G.remove_nodes_from(list(nx.isolates(G)))
    pos = nx.circular_layout(G)

    nx.draw(G, pos= pos, with_labels = True, node_size = 40, font_size = 15, font_weight = 'bold', 
            node_color = 'green', ax = ax, edge_color ='green' )

def add_labels(ax, xlabel='X', ylabel='Y', zlabel='Z', title='', xlim = None, ylim = None, zlim = None,xticklabels = np.array([None]),
               yticklabels = np.array([None] ), xticks = [], yticks = [], legend = [], ylabel_params = {'fontsize':18},
               zlabel_params = {'fontsize':18}, 
               xlabel_params = {'fontsize':18}, 
               title_params = {'fontsize':26}):
  """
  This function add labels, titles, limits, etc. to figures;
  Inputs:
      ax      = the subplot to edit
      xlabel  = xlabel
      ylabel  = ylabel
      zlabel  = zlabel (if the figure is 2d please define zlabel = None)
      etc.
  """
  if xlabel != '' and xlabel != None: ax.set_xlabel(xlabel, **xlabel_params)
  if ylabel != '' and ylabel != None:ax.set_ylabel(ylabel, **ylabel_params)
  if zlabel != '' and zlabel != None:ax.set_zlabel(zlabel,**ylabel_params)
  if title != '' and title != None: ax.set_title(title, **title_params)
  if xlim != None: ax.set_xlim(xlim)
  if ylim != None: ax.set_ylim(ylim)
  if zlim != None: ax.set_zlim(zlim)
  
  if (np.array(xticklabels) != None).any(): 
      if len(xticks) == 0: xticks = np.arange(len(xticklabels))
      ax.set_xticks(xticks);
      ax.set_xticklabels(xticklabels);
  if (np.array(yticklabels) != None).any(): 
      if len(yticks) == 0: yticks = np.arange(len(yticklabels)) +0.5
      ax.set_yticks(yticks);
      ax.set_yticklabels(yticklabels);
  if len(legend)       > 0:  ax.legend(legend)


def remove_edges(ax, include_ticks = False, top = False, right = False, bottom = False, left = False):

    """
    Removes selected edges and ticks from a Matplotlib axis object.

    Parameters:
        ax (matplotlib.axes.Axes): The axis object to modify.
        include_ticks (bool): If True, ticks will not be removed from the axis. Default is False.
        top (bool): If True, the top edge of the axis will be removed. Default is False.
        right (bool): If True, the right edge of the axis will be removed. Default is False.
        bottom (bool): If True, the bottom edge of the axis will be removed. Default is False.
        left (bool): If True, the left edge of the axis will be removed. Default is False.

    Returns:
        None

    Example usage:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        # Draw a plot on the axis object
        remove_edges(ax, include_ticks=False, top=True, right=True, bottom=True, left=True)
    """    
    ax.spines['top'].set_visible(top)    
    ax.spines['right'].set_visible(right)
    ax.spines['bottom'].set_visible(bottom)
    ax.spines['left'].set_visible(left)  
    if not include_ticks:
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])



def create_colors(len_colors, perm = [0,1,2], axis_keep = [], axis_vals = [0,0,0], min_vals = [0,0,0],max_vals = [1,1,1]):
    """
    Create a set of discrete colors with a one-directional order
    Input: 
        len_colors = number of different colors needed
    Output:
        3 X len_colors matrix decpiting the colors in the cols
    """
    colors = np.vstack([max_vals[0]*np.linspace(min_vals[0],1,len_colors),max_vals[1]*(1-np.linspace(min_vals[1],1,len_colors))**2,max_vals[2]*(1-np.linspace(0,1-min_vals[2],len_colors))])
    colors = colors[perm, :]
    if not checkEmptyList(axis_keep):
        for axis in axis_keep:
            colors[axis] = axis_vals[axis]
        
        
    return colors

"""
OPEN VOLTAGE
"""    
import tifffile
def open_voltage_penic(path_data = r'E:\CODES FROM GITHUB\GraFT_Python\datasets\graft_voltage\Penicillium', 
                       name_file = r'penicillium_high_snr.tif',
                       name_array = 'peni_data_array_npy.npy', to_save = False,
                       xmin = 300, xmax = 800,          ymin =  0, ymax = 'n', 
                       name_type ='VI_Peni' , use_array = False):#'CI_tele_zebra'
    """
    This function opens a voltage dataset stored as a TIFF file or a numpy array. The dataset can be a 
    Penicillium image stack or a zebrafish image stack. If the image stack is not stored as a numpy array, 
    it is read from the TIFF file. If to_save is True, the image stack is saved as a numpy array. 
    
    Parameters:
    -----------
    path_data : str
        The path to the directory containing the dataset file.
    name_file : str
        The name of the dataset file.
    name_array : str
        The name of the numpy array file containing the dataset. This parameter is used only if the image stack 
        is already stored as a numpy array.
    to_save : bool, default False
        Whether to save the image stack as a numpy array.
    xmin, xmax, ymin, ymax : int, default xmin=300, xmax=800, ymin=0, ymax='n'
        The boundaries of the image stack to be loaded. If ymax is 'n', the maximum value along the 
        z-dimension is used.
    name_type : str, default 'CI_tele_zebra'
        The name of the numpy array file to be saved. This parameter is used only if to_save is True.
    
    Returns:
    --------
    image_stack : numpy.ndarray
        The image stack representing the voltage dataset.
    
    Example:
    --------
    # Load Penicillium dataset
    path_data = r'E:\CODES FROM GITHUB\GraFT_Python\datasets\graft_voltage\Penicillium'
    name_file = r'penicillium_high_snr.tif'
    name_array = 'peni_data_array_npy.npy'
    xmin = 300
    xmax = 800
    ymin = 0
    ymax = 'n'
    name_type = 'CI_tele_zebra'

    image_stack = open_voltage_penic(path_data, name_file, name_array, to_save=False, 
                                     xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, name_type=name_type)
    
     # 1024 X 1024 X 1024
     # the following files were taken from https://zenodo.org/record/7330257#.ZDLVcXbMJhE for this paper
     # https://www.biorxiv.org/content/10.1101/2022.11.17.516709v1.full.pdf
     
     for VI peni:
         
         (default)
        
    for  VI zebra:
        
        (future)
     
     for CI zebra:
         path_data can be also r'E:\CODES FROM GITHUB\GraFT_Python\datasets\graft_voltage\Zebrafish' 
         name_array = 'data_array_npy_ci_dorsal_telencephalon.npy'
         save_name = 'CI_tele_zebra'
         np.save(create_data_name('CI_tele_zebra', xmin = 300, xmax = 800,
                                  ymin =  0, ymax = 'n'), np.transpose(image_stack[:,:400,300:800],[2,1,0]))
         np.save(create_data_name('CI_tele_zebra', xmin = 300, xmax = 800, ymin =  0, ymax = 'n')[:-4]+'info.npy', {'path':file_path, 'source': r"https://www.biorxiv.org/content/10.1101/2022.11.17.516709v1", 'data': np.transpose(image_stack[:,:400,300:800],[2,1,0])})
     """

    try:
        image_stack = np.load(path_data + os.sep + name_array, image_stack)
        used_array = True
    except: 
        file_path = path_data + os.sep + name_file
        with tifffile.TiffFile(file_path) as tiff:
            # read the 3D image stack
            image_stack = tiff.asarray()
        used_array = False
    if to_save:
        if use_array:
            print('not saving since used array')
        else:
            if len(name_type) == 0:
                name_type = name_file.split('.')  # 'CI_tele_zebra'
            full_name = create_data_name(name_type, xmin = xmin, xmax = xmax, 
                                     ymin =  ymin, ymax = ymax)
            np.save(full_name,     np.transpose(image_stack[:,ymin:ymax,xmin:xmax],[2,1,0]))
    return image_stack

    
    
from PIL import Image, ImageDraw,ImageEnhance


def create_gif(base, duration = 1, path_save = r'E:\CODES FROM GITHUB\GraFT_Python\datasets\graft_voltage\Penicillium',
               name_save = 'gif_try', to_normalize = False, to_return = False):
    """
    Creates a gif from a 3D array of base images and saves it to the specified location.
    
    Parameters:
        base (np.ndarray): The 3D array of images to create the gif from.
        duration (int, optional): The duration of each image in the gif. Defaults to 1.
        path_save (str, optional): The path to save the gif file. Defaults to "E:\CODES FROM GITHUB\GraFT_Python\datasets\synthetic_gif".
        name_save (str, optional): The name to save the gif file with. Defaults to "gif_synthetic".
        
    PAY ATTENTION!!!! the assumption here is that T is the 2nnd axis. If it is another axis (e.g. 0)
    then you should call:
        create_gif(np.transpose(image_stack,[2,1,0]))
        example: create_gif(np.transpose(image_stack[:400,:400,:400],[2,1,0]))
    """    
    if to_normalize:
        images = []
        for t in range(base.shape[2]):
            base_t = base[:,:,t]            
            high_perc = np.percentile(base_t,98)
            low_perc = np.percentile(base_t,2)
            base_t_norm = (base_t - low_perc)/(high_perc - low_perc)
            base_t_norm[base_t_norm > 1] = 1
            base_t_norm[base_t_norm < 0] = 0
            images.append(Image.fromarray((base_t_norm*255).astype(np.uint8)))
            
    else:        
        #perc_val = np.percentile(image_stack[0,:,:],95)
        images = [Image.fromarray((base[:,:,t]*16).astype(np.uint8)) for t in range(base.shape[2])]
    images[0].save(path_save + os.sep +name_save +'.gif', save_all=True, 
                   append_images=images[1:], duration=duration)
    if to_return:
        return images
    

def remove_background(ax, grid = False, axis_off = True):
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    if not grid:
        ax.grid(grid)
    if axis_off:
        ax.set_axis_off()
    
        
def create_ax(ax, nums = (1,1), size = (10,10), proj = 'd2',return_fig = False,sharey = False, sharex = False, fig = []):
    #ax_copy = ax.copy()
    if isinstance(ax, list) and len(ax) == 0:
        #print('inside')
        if proj == 'd2':
            fig,ax = plt.subplots(nums[0], nums[1], figsize = size, sharey = sharey, sharex = sharex)
        elif proj == 'd3':
            fig,ax = plt.subplots(nums[0], nums[1], figsize = size,subplot_kw={'projection':'3d'}, sharey = sharey, sharex = sharex)
        else:
            raise NameError('Invalid proj input')
        if return_fig:
            return fig, ax

    if  return_fig :
        return fig, ax
    return ax

def plot_3d(mat, ax, lw = 2, color ='black', ls = '-', alpha = 1):
    ax.plot(mat[0],mat[1],mat[2], color = color, lw = lw, ls = ls, alpha = alpha)


def open_word2vec(word2vec_source = 'word2vec-google-news-300', word_list = 'google_trends', to_save = True):
    # BETTER TO RUN OUTSIDE OF FUNcTION!!!!!!!!!!!!!!!!!!!
    import gensim.downloader as api
    # load pre-trained Word2Vec model
    model = api.load(word2vec_source)

    # define a list of strings
    
    if isinstance(word_list, str):
        if word_list.lower() == 'google_trends':
            dict_trends = np.load('grannet_trends_results_march_2023.npy', allow_pickle=True).item()
            word_list = np.array(dict_trends['df_dict']['CA'].keys()).astype(str)
    #word_list = ['apple', 'banana', 'orange']

    # get embeddings for words in the list
    embeddings_full =[]
    store_used_words = []
    store_split = []

    for word in word_list:
        try: 
            embeddings_full = vstack_f(embeddings_full, model[word])
        except(KeyError):
            try:
                embeddings_full = vstack_f(embeddings_full, model[word.split()[0]])
                store_split.append(word); word = word.split()[0]
            except:
                print('could not find word %s'%word)

        store_used_words.append(word)
        
    #embeddings = [model[word] for word in word_list]

    # print the embeddings
    #print(embeddings)
    if to_save:
        np.save('grannet_trends_words_embed_2023.npy', {'store_used_words':store_used_words,
                                                        'emb':np.vstack(embeddings_full),
                                                        'store_split': store_split})
    return embeddings

def hypo_test(data, dist = 'poi')    :
    if dist.startswith('poi'):
        from scipy.stats import chisquare, poisson

    # Generate some sample data
    T = 1000
    data = np.random.poisson(5, T)
    
    # Calculate the observed frequencies
    observed, _ = np.histogram(data, bins=np.arange(np.max(data)+2))
    
    # Calculate the expected frequencies under the Poisson distribution
    rate = np.mean(data)
    expected = poisson.pmf(np.arange(np.max(data)+1), rate) * T
    
    # Perform the chi-square goodness-of-fit test
    _, p_value = chisquare(observed, expected, ddof=1)
    
    # Print the p-value
    print("p-value:", p_value)
        
def vstack_f(ar1, ar2, direction = 0)    :
    
    """
    Stack arrays vertically including initialization
    
    Parameters:
    ar1 (numpy.ndarray): The first array to stack.
    ar2 (numpy.ndarray): The second array to stack.
    direction (int, optional): The direction in which to stack the arrays (default=0).
    
    Returns:
    numpy.ndarray: The stacked array.
    
    Raises:
    ValueError: If the shapes of the input arrays are not consistent.
    
    Example:
    >>> a = np.array([[1, 2], [3, 4]])
    >>> b = np.array([[5, 6]])
    >>> vstack_f(a, b)
    array([[1, 2],
           [3, 4],
           [5, 6]])
    """
    if direction == 0:
        if len(ar2.flatten()) == np.max(ar2.shape):
            ar2 = ar2.reshape((1,-1))
        if checkEmptyList(ar1):
            return ar2
        else:
            if len(ar1.flatten()) == np.max(ar1.shape):
                ar1 = ar1.reshape((1,-1))
            if ar1.shape[1]  != ar2.shape[1]:
                raise ValueError('shapes are not consistent! shape ar1 is %s and shape ar2 is %s'%(str(ar1.shape), str(ar2.shape)))
            return np.vstack([ar1,ar2])
    elif direction == 1:
        if checkEmptyList(ar1):
            a1_T = []
        else:
            a1_T = ar1.T
        return vstack_f(a1_T, ar2.T, 0).T 
    
    
        
    
    
    

# to plot hierarchical clustering dendogram 
# plt.figure(); sns.clustermap(df_data)



#file_path = r'E:\CODES FROM GITHUB\GraFT_Python\datasets\graft_voltage\Zebrafish' + os.sep + 'zebrafish_Dorsal_telencephalon.tif'



# print the shape of the image stack
# print(image_stack.shape)
# with tifffile.TiffFile(file_path) as tiff:
#     # read the 3D image stack
#     image_stack = tiff.asarray()



#%%  MATRIX FACTORIZATION METHODS FOR COMPARISON
"""""""""""
HERE - WRITE METHODS FOR OTHER MATRIX FACTORIZATION METHODS FOR COMPARISON
"""""""""""
try:
    from fcmeans import FCM
except:
    print('did not import fcmeans')

def cluster_trends(type_cluster = 'emb', trends_path = '' , p = 2, m = 2, to_fuzzy = True, A = []):
    # type_cluster can be 'emb' or ''
   # if trends_data    
    if type_cluster == 'emb':
        if len(trends_path) == 0:
            trends_path =  r'grannet_trends_words_embed_2023.npy'
        trends_data = np.load(trends_path, allow_pickle = True).items()
        X = trends_data['emb']
        words_used = trends_data['store_used_words']
        fcm_labels, W = run_fuzzy_clustering(X, p, m,  True, A)    
    elif type_cluster == 'temporal':
        if len(trends_path) == 0:
            trends_path = r'grannet_trends_results_march_2023.npy'
        dict_trends = np.load(trends_path, allow_pickle=True).item()
        word_list = np.array(dict_trends['df_dict']['CA'].keys()).astype(str)
    
    
def fuzzy_clasification_give_states_dict(df_dict = [], type_fuzzy = 'single'):
    # to apply fuzzy clustering
    # type_fuzzy can be single or together
    if checkEmptyList(df_dict):
        print('pay attention! df_dict not give, upload trends vals')
        trends_path = r'grannet_trends_results_march_2023.npy'
        dict_trends = np.load(trends_path, allow_pickle=True).item()
        #word_list = np.array(dict_trends['df_dict']['CA'].keys()).astype(str)
    if type_fuzy == 'single':
        W_return = {}
        for key in list(df_dict.key()):                
            _, W = run_fuzzy_clustering(X, p = 3 ,m = 2, to_fuzzy = True, A = [])
            W_return['key'] = W
    elif type_fuzy == 'together':
        stack_vals = np.hstack([item for _,item in df_dict.items()])
        _, W_return = run_fuzzy_clustering(X, p = 3 ,m = 2, to_fuzzy = True, A = [])
            
    return W_return 
            
    
    
    
    
def run_fuzzy_clustering(X, p = 3 ,m = 2, to_fuzzy = True, A = []):
    """
    data should be samples X features (neurons X time)
    m= how fuzzy
    p = num clusters
    using the code from https://pypi.org/project/fuzzy-c-means/
    explanation here https://fuzzy-c-means.readthedocs.io/en/latest/examples/00%20-%20Basic%20clustering/
    paper here https://www.sciencedirect.com/science/article/pii/0098300484900207?via=ihub
    centers are num centers x dimenions of X
    """
    fcm = FCM(n_clusters=p, m = m)
    fcm.fit(X)    
    fcm_centers = fcm.centers    
    fcm_labels = fcm.predict(X)   
    
    if to_fuzzy:
        num_samples = X.shape[0]
        W = np.zeros((num_samples, p))
        for center in  range(p):
            cj = fcm_centers[center,:]
            for sample_num in range(num_samples):
                xi = X[sample_num]
                
                d = claculate_d_fuzzy(xi, cj, A )  
                nominator = d**(2/(m-1))
                d_powers_deno = []
                for center_deno in range(p):
                    ck = fcm_centers[center_deno,:]
                    d_deno = claculate_d_fuzzy(xi, ck, A )  
                    d_powers_deno.append(nominator / d_deno**(2/(m-1)))
                full_deno = np.sum(d_powers_deno)
                W[sample_num, center] = 1/full_deno
            
        return fcm_labels, W
    return fcm_labels
    
    
def claculate_d_fuzzy(vec1, vec2, A = [])  :
    """
    Calculates the fuzzy distance between two vectors, vec1 and vec2, using a
    user-defined weight matrix A. If no weight matrix is provided, the identity
    matrix will be used.
    
    Parameters:
    vec1 (numpy.ndarray): The first vector.
    vec2 (numpy.ndarray): The second vector.
    A (numpy.ndarray): The weight matrix. Default is the identity matrix.
    
    Returns:
    d (float): The fuzzy distance between vec1 and vec2.
    """
    # paper page 3
    # https://www.sciencedirect.com/science/article/pii/0098300484900207?via=ihub
    l = len(vec1.flatten())
    vec1 = vec1.flatten()
    vec2 = vec2.flatten()
    if checkEmptyList(A):
        A = np.eye(l)
    d  = np.square( (vec1-vec2).reshape((1,-1)) @ A @ (vec1-vec2).reshape((-1,1)))
    return d



"""
existing methods

"""
from tensorly.decomposition import tucker, parafac, non_negative_tucker
from tensorly import random as random_tl

def run_existing_methods(data, p, methods_to_compare = ['adad_svd','hosvd','parafac','tucker', 'HOOI'],
                         params_parafac = {}, params_tucker = {}):
    # the mathods are taken from http://tensorly.org/stable/modules/api.html#module-tensorly.decomposition
    # user guide http://tensorly.org/stable/user_guide/quickstart.html#tensor-decomposition
    results = {}
    A_tucker, phi_tucker, _,_ = run_tucker(data, p = p, params_tucker = params_tucker)
    results['tucker'] = {'A':A_tucker, 'phi':phi_tucker}
    A_parafac, phi_parafac, _ = run_parafac(data, p = p, params_parafac = params_parafac)
    results['prafac'] = {'A':A_parafac, 'phi':phi_parafac}
    return results
    
    
def run_adad_svd(data, p = 10, max_TK = 1000):  
    """
    Runs the adaptive SVD decomposition on the given data.

    Parameters:
        data (ndarray): Input data array.
        p (int): Number of components to keep.
        max_TK (int): Maximum number of trials.

    Returns:
        A (ndarray): Matrix A from the decomposition.
        phi (ndarray): Matrix phi from the decomposition.
    """    
    A, s, VT = np.linalg.svd(data[:,:,0])
    A = A[:p]
    phi = VT
    return A, phi
    
    
    
    
def run_tucker(data, p = 10, params_tucker = {}):
    """
    Runs the Tucker decomposition on the given data.

    Parameters:
        data (ndarray): Input data array.
        p (int): Number of components to keep.
        params_tucker (dict): Additional parameters for the Tucker decomposition.

    Returns:
        A_tucker (ndarray): Matrix A from the Tucker decomposition.
        phi_tucker (ndarray): Matrix phi from the Tucker decomposition.
        core (ndarray): Core tensor from the Tucker decomposition.
        factors (tuple): Factors of the Tucker decomposition.
    """
    k = data.shape[2]
    T = data.shape[1]
    core, factors = tucker(data, rank=[p, p, T], ** params_tucker)
    A_tucker = factors[0]
    phi_base_tucker = factors[1]
    k_tucker = factors[2]
    

    phi_tucker = np.dstack([phi_base_tucker*k_tucker[k_spec, k_spec] for k_spec in range(k)])   
    
    return A_tucker, phi_tucker, core, factors
    
    
    
def run_parafac(data, p = 10, params_parafac = {}):   
    """
    decomposition: http://tensorly.org/stable/modules/generated/tensorly.decomposition.parafac.html#tensorly.decomposition.parafac
    parafac paper: https://www.cs.cmu.edu/~pmuthuku/mlsp_page/lectures/Parafac.pdf
    """
    N = data.shape[0]
    T = data.shape[1]
    k = data.shape[2]
    # gives me a list of N x p; T x p; k x p
    #np.random.rand(N,T,k)
    factors = parafac(data, rank=p, **params_parafac)
    factors_f = factors.factors
    factors_w = factors.weights
    
    full_A = []
    full_phi = []
    
    A_parafac = factors_f[0]
    phi_base_parafac = factors_f[1]
    print('phi_base_parafac')
    print(phi_base_parafac.shape)
    k_parafac = factors_f[2]
    for k_spec in range(k):         
        phi_parfac = np.dstack([phi_base_parafac*k_parafac[k_spec,:].reshape((1,-1)) for k_spec in range(k)] )
        
    return A_parafac, phi_parfac, factors
    
    
    
  
    




