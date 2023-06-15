# -*- coding: utf-8 -*-
"""

@author: Noga Mudrik
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 23:27:40 2022

@author:---
NOTE: THE FORMER NAME OF OUR MODEL WAS GRANNET (instead of SiBBlInGS). While we decided to change the model name to SiBBlInGS as it better captures the model goal,
here we still call it grannet. 
Hence, wherever there is a mention of "grannet", we refer to our model, SiBBlInGS

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
"""
example
"""
# example: run_GraFT(data =[], corr_kern = [],  params = params_default, grannet=False)
"""
Imports
"""
from sklearn.neighbors import NearestNeighbors
import os
import mat73
import scipy.io as sio
from scipy.optimize import nnls
from sklearn.decomposition import PCA

from qpsolvers import solve_qp # FROM https://scaron.info/doc/qpsolvers/quadratic-programming.html#qpsolvers.solve_qp https://pypi.org/project/qpsolvers/
import matplotlib
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
from skimage import io
global ask_selected
import networkx as nx
from datetime import datetime as datetime2

ask_selected = False
in_local = True    
   
ss = int(str(datetime2.now()).split('.')[-1])
seed = ss 
np.random.seed(seed)

    


def str2bool(str_to_change):
    """
    Converts a string representation of a boolean value to a boolean variable.

    Parameters:
        str_to_change (str): String representation of a boolean value.

    Returns:
        bool: Converted boolean value.

    Example:
        str2bool('true') -> True
    """
    if isinstance(str_to_change, str):
        str_to_change = (str_to_change.lower()  == 'true') or (str_to_change.lower()  == 'yes') or (str_to_change.lower()  == 't')
    return str_to_change

 
"""
Default Parameters 
"""
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
link of regularization to paper:
    code - paper
    l1 - lambda
    l2 - gamma 1 (frob norm)
    l3 - gamma 3 continuation
    l4 - gamma 2 (diag - correlations)
    l5 - gamma 4 time continues

important - to set the parameters choose type_answer = 1
"""
type_answer = 1


##########################################################
"""
IMPORTANT PARAMETERS
'epsilon' = nominator 
beta = free element
lambdas = params['epsilon']/(beta + A + H @ A) 
"""
###########################################################
if type_answer not in [0,1]: raise ValueError('undefined type_answer')

def decide_value(type_answer, text, val, type_choose = 'input'):
    if type_answer == 0:
        return val
    else:
        if type_choose == 'input':
            return input(text)
        else:
            return text

params_default = {'max_learn': 1e4,                                                                 # Maximum number of steps in learning 
    'mean_square_error': 0.05,
    'deal_nonneg': 'make_nonneg',                                                                    # can be 'make_nonneg' or 'GD'
    'epsilon' : decide_value(1, 0.2, np.random.rand()*0.1+0.1, 'val'),                               # Default values to be spatially varying
    'zeta' : decide_value(1, 2, np.random.rand()*0.3+ 0.9, 'val'),                                   # weight of graph in labdas (w_graph)
    'l1': decide_value(type_answer, 1, np.random.rand(), 'val'),                                     # the whole llambda component weighting
    'l2': decide_value(type_answer, 0, np.random.rand(), 'val'),                                     # Default Forbenius norm parameter is 0 (don't use)
    'l3': decide_value(type_answer, 0, np.random.rand(), 'val'),                                     # Default Dictionary continuation term parameter is 0 (don't use)
    'l5':  decide_value(type_answer, 0.2, np.random.rand(), 'val'),                                  # smoothness over time 
    'lamContStp': decide_value(type_answer, 0.1, np.random.rand(), 'val')*0.9,                       # Default multiplicative change to continuation parameter is 1 (no change)
    'l4': decide_value(type_answer, 0.1, np.random.rand(), 'val')*0.1,                               # Default Dictionary correlation regularization parameter is 0 (don't use)
    'beta': 0.9*decide_value(0, 0.1, np.random.rand()*0.1+0.05, 'val'),                              # Default beta parameter to 0.09
    'maxiter': 0.01,                                                                                 # Default the maximum iteration to whenever Delta(Dictionary)<0.01
    'numreps': 5,                                                                                    # Default number of repetitions for RWL1 is 2
    'tolerance': 1e-7*decide_value(type_answer, 0.5, 0.8*np.random.rand(), 'val'),                   # Default tolerance for TFOCS calls is 1e-8
    'likely_from' : decide_value(type_answer, 'gaussian' , 'gaussian', 'val') ,                      # Default to a gaussian likelihood ('gaussian' or'poisson')
    'step_s': decide_value(type_answer, 0.2, 0.1+0.9*np.random.rand(), 'val'),                       # Default step to reduce the step size over time                                                                       
    'step_decay': np.min([0.9999,0.995 + 0.005*decide_value(type_answer, 0.9, np.random.rand(), 'val')]),     # Default step size decay (only needed for grad_type ='norm')
    'dict_max_error': 0.01,        # Default learning tolerance: stop when Delta(Dictionary)<0.01
    'p': 7,                        # Default number of dictionary elements is a function of the data
    'verb': 1,                     # Default to no verbose output    
    'GD_iters': 1* decide_value(type_answer, 1, np.random.randint(1,5), 'val'),                      # Default to one GD step per iteration
    'nonneg': decide_value(type_answer, False, False,'val')   ,  #                                   # Default to not having negativity constraints on the coefficients
    'plot': False,                                                                                   # Default to not plot spatial components during the learning
    'updateEmbed' : False,                                           # Default to not updateing the graph embedding based on changes to the coefficients
    'normalizeSpatial' : False,                                      # default behavior - time-traces are unit norm. when true, spatial maps normalized to max one and time-traces are not normalized     
    'reduceDim': decide_value(type_answer, False, np.random.choice([False, True]), 'val'),           # reduce dim
     'n_neighbors': decide_value(type_answer,49, np.random.randint(5,50), 'val'),                    # k neighbors fo r the graph building    
     'n_comps':5,                                                                                    # number of components of reduced dim
     'solver_qp':'quadprog',                                                                         #solver updating A for nonneg
     'solver': decide_value(0 ,'lasso', np.random.choice(['spgl1','inv', 'lasso']), 'val'),          # solver for A 
     'nullify_some': False ,                                                                         # if to nullify very small values of A       
     'norm_by_lambdas_vec': decide_value(type_answer,True, np.random.choice([False, True]), 'val'),  # if to apply weighted LASSO (true) or regular LASSO (false)
     'min_max_data': False,                                                                          # if to apply min max normalization
     'GD_type': 'full_ls_cor',                                                                       # which regulazriation on phi to include. full_ls_cor is the full version
     'thresANullify':-50,                                                                            # percentile of low values of A to nullify if nullify_some is True
     'name_addition'  : '',                                                                          # name for saving                 
     'one_inv': False,                                                                               # if to include an inverse step for phi every few iterations
     'add_noise_small_A':False,                                                                      # if to perturb A when the error is stucked
     'init_same':True,                                                                               # if to initialize A the same for all states
     'T_inv':2,                                                                                      # if 'one_inv', then 'T_inv' decides in how many iterations the inverse will takeplace
   
     'CI': {                                                                                         # parameters if applying to calcium imaging data 
     'xmin' : 151,#151
     'xmax' : 200,#350                                                           
     'ymin' : 151,#101
     'ymax' : 200,#300 
     },
     'VI_crop': {                                                                                     # parameters if applying to voltage imaging data    
     'xmin' : 120,#151
     'xmax' : 270,#350                                                            
     'ymin' : 0,#101
     'ymax' : -1,#300 
     },
     'VI_crop_very': {                                                                                # parameters if applying to voltage imaging data  
     'xmin' : 120,#151
     'xmax' : 170,#350                                                            
     'ymin' : 20,#101
     'ymax' : 70,#300 
     },     
     
    'synth_grannet':                                                                                   # parameters if applying to  synthetic data for sibblings
        {
    'xmin' : 0,#151
    'xmax' : 'n',#350                                                           
    'ymin' : 0,#101
    'ymax' : 'n',#300 
            },
    
    'trends_grannet' :                                                                                # parameters if applying to Google trends data for sibblings
    {
    'xmin' : 0,#151
    'xmax' : 'n',#350                                                          
    'ymin' : 0,#101
    'ymax' : 'n',#300 
            },
    'neuro_bump_angle_active_minmax' :                                                                # parameters if applying to neural data for sibblings
    {
    'xmin' : 0,#151
    'xmax' : 'n',#350                                                            
    'ymin' : 0,#101
    'ymax' : 'n',#300 
            },     
     'use_former_kernel' : False,                                                                     # if to use  a pre-calculate kernel to save time
     'usePatch' : False,   
     'divide_med' : False,                                                                            # if to divide the data by the data median as pre-processing
     'to_save' : True,                                                                                # if to save the results
     'default_path':  r'.',                                                                           # default path for data 
     'save_error_iterations': True,                                                                   # if to save error during iterations
     'max_images':800,                                                                                #  maximum time point
     'dist_init': 'rand',                                                                             # distribution for matrix initialization
     'to_sqrt':True,                                                                                  # if to square initialized matrix
     'sigma_noise': 0.1,                                                                              # std of noise to add to dict if all values are zeros
     'grannet_params':{'lambda_grannet': 0.1,                                                         # parameters specifically related to sibblings
                       'distance_metric':'Euclidean',
                       'reg_different_nets': 'unified',
                       'num_free_nets':0,
                       'distance2proximity_trans': 'exp',
                       'sigma_distance2proximity_tran':10,
                       'initialize_by_other_nets':True,                                               # True if to initialize A by the other nets
                       'late_start': 0,
                       'include_Ai':False,                                                            # whether to include the distance from Ai to itself in NeuroInfer (if True - kind of smoothness regularization)
                       'labels_indicative':False ,
                       'rounded': False,
                       'rounded_max': 360},                                                           # are the labels closing a loop? i.e. like angles?1 if so - aaume \delta(lab1, lab2) == \delta(lab1, lab-1)
     'to_store_lambdas':False,                                                                        # if to save lambdas / store them
     'reorder_nets_by_importance': False,                                                             # if to reorder_nets_by_importance             
     'uniform_vals': [0,1],                                                                           # min and max values for matrix init. (uniform dist.)
     'distance_metric': 'euclidean' ,                                                                 # distance_metric for kernels
     'save_plots': True,                                                                              # if to save plots
      'graph_params':{'kernel_grannet_type':'combination', 'params_weights': 0.4, 'increase_sim': 1}, # params_weights should be a vector if kernel_grannet_type is averaged and a scalar if combined
      'hard_thres' : True,                                                                            # if to take hard thres on A
      'hard_thres_params': {'non_zeros': 21, 'thres_error_hard_thres': 3, 'T_hard': 1,   'keep_last':False}, # params for hard thres on A
      'add_inverse': True,                                                                            # if to add inverse on A
      'inverse_params': {'T_inverse':2},                                                              # inverse params on A
      'compare_to_synth': True,                                                                       # if to compare to ground truth (only for synthetic)
      'various_steps': True,                                                                          # if to check multiple steps for GD
      'steps_range': np.array([1, 1.1]) ,                                                             # in case of GD, range of options for changing the GD step 
      'step_loc': 1,                                                                                  # choosing the step-loc-th best option
      'phi_only_dec':True,                                                                            # only take update step on phi if error decreases
      'A_only_dec':False,                                                                             # only take update step on phi if error decreases
      'min_add_phi_error': 0.05,                                                                      # parameter for phi error decrease
      'T_save_fig': 10,                                                                               # time period for saving figure
      'is_trials': False,                                                                             # whether the data is multi trial case
      'is_trials_type': 'shared',                                                                     # is trials type can be shared = shared kernel, shared ners / shared_flex - shared kernel, flexible nets / flex - different kernels, different nets
      'num_nets_sample_trials': 0 ,                                                                   # how many networks to consider in a batch? (apply in the future!!)
      'add_noise_stuck':  True,                                                                       # if to perturb if error is stucked     
      'noise_stuck_params': {'max_change_ratio': 0.05, 'in_a_row': 5, 'std_noise_A':0.05, 'std_noise_phi': 0.3, 'change_step': 20 },  # in_a_row = how many times  the error remains; max_change_ratio - ratio of error that is considered unchanged
      'condition_unsupervised': True,                                                                 # if True - data driven P. if False - supervised P.
      'to_norm_cond_graph':False,                                                                     # if to norm the states graph H
      'min_step_size': 1e-10,                                                                         # minimum step size
      'max_step_size': 50,                                                                            # maximum step size
      'addition_norm_cond':1,                                                                         # how much to add to the normalization of similar nets
      'drop_med': False,                                                                              # substuct the median from each ste observation                                                         
      'add_phi_smooth': False,                                                                        # if to add a moving avg step over time for phi                                                
      'wind':5,                                                                                       # window for moving avg
      'weight_sim_nets': 3,                                                                           # how much weight to give to P
      'norm_A' :     True,                                                                            # if to norm A (axis 0)
      'norm_A_cols' :False,                                                                           # if to norm A (axis 1)
      'norm_A_params':{'type_norm_A': 'min_max' },                                                    # parameters for normalization. options for now are only min_max
      'boots_trials':True,                                                                            # if to include bootstrap of trials
      'boots_size' : 10,                                                                              # bootstrap size
      'initial_thres' : False,                                                                        # if to initialize A as a sparse matrix                                                                
      'phi_positive' : False,                                                                         # force positivity on phi
      'norm_by_sum':True,                                                                             # make sure than each channel's sum if fixed  
      'params_spgl1':{'decay':0.99}                                                                   # parameters for spgl1 solver
          
      
      
    }
if  params_default['likely_from'].lower() not in ['gaussian','poisson']:
    raise ValueError('likely_from must be poisson or gaussian')

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
check params
"""
if params_default['graph_params']['kernel_grannet_type'] not in ["one_kernel", "averaged", "combination", "ind" ]:
    raise ValueError('invalid kernel_grannet_type in graph_params (current %s)'%params_default['graph_params']['kernel_grannet_type'])
    

if params_default['is_trials'] and params_default['is_trials_type'] not in ['shared' , 'shared_flex', 'flex']:
    raise ValueError("invalid is_trials_type, shoulbe be in ['shared' , 'shared_flex', 'flex'] but %s"%params_default['is_trials_type'] )




global data_types
data_types  = ['CI', 'VI_crop', 'VI_full', 'VI_crop_very', 'area2', 'trends', 'VI_HPC', 'synth_trials_grannet','VI_Peni_partt',
               'VI_crop_long', 'VI_crop_long2','synth','synth_grannet','trends_grannet','neuro_bump_short','neuro_bump_short_short', 'neuro_bump_angle_active','neuro_bump_angle_active_minmax',
               'VI_Peni', 'ephy2_partt','ephy_partt', 'VI_Peni_partt', 'EEG']



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
suBBlInGS functions
"""

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
    if return True -> stuck
    if return false - >not stuck but decrease
    """

    if len(last_errors) > params_stuck['in_a_row']:
        last_errors =  last_errors[-params_stuck['in_a_row']:]
    max_thres = params_stuck['max_change_ratio']*last_errors[-1]
    return (np.abs(np.diff(last_errors)) < max_thres).all()
    
def add_noise_if_stuck(last_errors, A, phi, step_GD,  params_stuck) :
    """
    Converts labels to numerical values and creates a dictionary mapping label numbers to labels.

    Parameters:
        labels (list): List of labels.

    Returns:
        dict_nums_labels (dict): Dictionary mapping label numbers to labels.
        list_nums (list): List of numerical label values.
    """    
    if check_error_stuck(last_errors, params_stuck) :
        
        ss = int(str(datetime2.now()).split('.')[-1])
        np.random.seed(ss)
        A = A + np.random.randn(*A.shape)*params_stuck['std_noise_A']
        phi = phi + np.random.randn(*phi.shape)*params_stuck['std_noise_A']
        step_GD *= params_stuck['change_step']
    return A,phi, step_GD
  


def labels_to_nums(labels):
    """
    Converts labels to numerical values and creates a dictionary mapping label numbers to labels.

    Parameters:
        labels (list): List of labels.

    Returns:
        dict_nums_labels (dict): Dictionary mapping label numbers to labels.
        list_nums (list): List of numerical label values.
    """    
    dict_nums_labels ={}
    list_nums = []
    for label_num, label in enumerate(labels):
        dict_nums_labels[label_num] = label
        list_nums.append(label_num)
    return dict_nums_labels, list_nums


def fine_pos_from_angle(angles):
    """
    Converts labels to proximity and distance matrices.

    Parameters:
        labels (ndarray): Array of labels.
        distance_metric (str): Distance metric to be used.
        distance2proximity_trans (str): Transformation function for converting distance to proximity.
        rounded (bool): Indicates if the labels should be rounded.
        rounded_max (int): Maximum value for rounded labels.
        params (dict): Additional parameters.

    Returns:
        distance (ndarray): Distance matrix.
        proximity (ndarray): Proximity matrix.
    """    
    sins = [np.sin(ang) for ang in angles]
    coss = [np.cos(ang) for ang in angles]
    return np.vstack([sins, coss])

    
def labels2proximity(labels, distance_metric = 'Euclidean', distance2proximity_trans = 'exp', rounded = False , 
                     rounded_max = 360, params = params_default):
    """
    Converts labels to proximity and distance matrices.

    Parameters:
        labels (ndarray): Array of labels.
        distance_metric (str): Distance metric to be used.
        distance2proximity_trans (str): Transformation function for converting distance to proximity.
        rounded (bool): Indicates if the labels should be rounded.
        rounded_max (int): Maximum value for rounded labels.
        params (dict): Additional parameters.

    Returns:
        distance (ndarray): Distance matrix.
        proximity (ndarray): Proximity matrix.
    """
    #if rounded:

    if np.max(np.shape(labels)) == len(labels.flatten()):
        distance_base = np.abs(labels.reshape((1,-1)) - labels.reshape((-1,1)))

    else: 
        distance_base = np.sqrt(np.sum( np.dstack([(labels[row,:].reshape((1,-1)) - labels[row,:].reshape((-1,1)))**2 for row in np.arange(labels.shape[0])]),2))

    distance_base =  distance_base +10**3*(np.eye( distance_base.shape[0]))
    distance_base =  distance_base - np.diag(np.diag( distance_base)) + 0.5*np.eye( distance_base.shape[0])*np.min( distance_base)

        
    if distance_metric == 'Euclidean':
        distance = distance_base ** 2
    elif distance_metric == 'abs':
        distance = np.abs(distance_metric)       
    else:
        raise ValueError('Unknown Distance Metric!')
    if distance2proximity_trans == 'exp':
        proximity = np.exp(-distance/params['grannet_params']['sigma_distance2proximity_tran'])

    elif  distance2proximity_trans == 'inv':
        proximity = 1/distance
    else:
        raise ValueError('Unknown Proximity Metric!')     
    
    return distance, proximity


def lists2list(xss)    :
    """
    Flattens a list of lists into a single list.

    Parameters:
    ----------
    xss : list of lists
        The input list of lists.

    Returns:
    -------
    list
        The flattened list.

    Examples:
    ---------
    >>> lists2list([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    [1, 2, 3, 4, 5, 6, 7, 8, 9]

    >>> lists2list([[1], [2, 3], [4, 5, 6]])
    [1, 2, 3, 4, 5, 6]
    """    
    return [x for xs in xss for x in xs] 

def create_proximity_coeffs_based_on_prxoimity(p, proximity = [], num_free_nets = 0, reg_different_nets = 'unified',
                                               num_states = 0, nu = [], params = params_default, data = [], unsup = True):
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
    
    condition_unsupervised = we want to use the distances. we do not know (inverse on inverse)
    """
    

    if params['condition_unsupervised'] and checkEmptyList(data): # UNSEUPRVISED!!! (data driven P)
        raise ValueError('data should not be emptylist if "condition_unsupervised"')
    elif checkEmptyList(proximity) and params['condition_unsupervised'] :
        proximity  = cal_dist(0, MovToD2_grannet(data.transpose((2,1,0))), graph_params = params['graph_params'],
                 grannet = True, distance_metric = params['distance_metric'])


    elif checkEmptyList(proximity):
        """
        if superrvsied but unknown labels = equal labels
        """
        proximity = np.ones((num_states, num_states, 1))
    else:
        print('completely supervised')
        
    if params['to_norm_cond_graph']:
        proximity = norm_vec_min_max(proximity) + params['addition_norm_cond']    
    if not checkEmptyList(nu):
        if len(nu) != p:
            raise ValueError('Len of $nu$ (%d) must be equal to the number of nets (%d)'%(len(nu), p))
        coeffs_mat = np.dstack([proximity*nu_i for nu_i in nu])
        #np.repeat(proximity.reshape((proximity.shape[0], proximity.shape[1],1)), p, axis = 2)
    else:

        if reg_different_nets == 'unified': # the kernel is the same
            coeffs_mat = np.dstack([proximity.reshape((proximity.shape[0], proximity.shape[1],1))]*p)

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
    if num_free_nets != 0:
        coeffs_mat[:,:,-num_free_nets:] = 0
    
    return coeffs_mat



def createDefaultParams(params = {}):
    """
    Creates a dictionary of default parameters by combining a default dictionary with additional parameters.

    Parameters:
    ----------
    params : dict, optional
        Additional parameters to be added or overwritten in the default dictionary. (default: {})

    Returns:
    -------
    dict
        The dictionary of default parameters.

    Examples:
    ---------
    >>> createDefaultParams()
    {'step_s': 1, 'learn_eps': 0.01, 'epsilon': 2, 'numreps': 2}

    >>> createDefaultParams({'learn_eps': 0.05, 'max_iters': 100})
    {'step_s': 1, 'learn_eps': 0.05, 'epsilon': 2, 'numreps': 2, 'max_iters': 100}
    """    
    dictionaryVals = {'step_s':1, 
                      'learn_eps':0.01,
                      'epsilon': 2,
                      'numreps': 2, 
                      }
    return  addKeyToDict(dictionaryVals,params)

def createLmabdasMat(epsilonVal, shapeMat):
    """
    Creates a matrix of lambdas based on the input epsilonVal and shapeMat.

    Parameters:
    ----------
    epsilonVal : int, float, list, tuple, or np.ndarray
        The value(s) used to create the lambdas matrix. If a single value is provided, it is used to populate the entire matrix.
        If a list, tuple, or numpy array is provided, the length of epsilonVal must be equal to either the number of rows or columns in shapeMat.

    shapeMat : tuple
        The shape of the lambdas matrix (rows, columns).

    Returns:
    -------
    np.ndarray
        The matrix of lambdas with shape shapeMat.

    Raises:
    ------
    ValueError - If epsilonVal is not a number or a list/tuple/np.ndarray with the number of elements equal to one of the shapeMat dimensions.

    Examples:
    ---------
    >>> createLmabdasMat(0.5, (3, 3))
    array([[0.5, 0.5, 0.5],
           [0.5, 0.5, 0.5],
           [0.5, 0.5, 0.5]])

    >>> createLmabdasMat([1, 2, 3], (4, 3))
    array([[1, 2, 3],
           [1, 2, 3],
           [1, 2, 3],
           [1, 2, 3]])

    >>> createLmabdasMat([0.1, 0.2, 0.3, 0.4], (2, 4))
    array([[0.1, 0.2, 0.3, 0.4],
           [0.1, 0.2, 0.3, 0.4]])
    """    
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
    """
    Combines two dictionaries into a single dictionary.

    Parameters:
    ----------
    dictionaryVals : dict
        The base dictionary.

    dictionaryPut : dict
        The dictionary to be added or overwritten in the base dictionary.

    Returns:
    -------
    dict
        The combined dictionary.

    Examples:
    ---------
    >>> addKeyToDict({'a': 1, 'b': 2}, {'c': 3, 'd': 4})
    {'a': 1, 'b': 2, 'c': 3, 'd': 4}

    >>> addKeyToDict({'a': 1, 'b': 2}, {'b': 3, 'c': 4})
    {'a': 1, 'b': 3, 'c': 4}
    """    
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
    """
    Plot two networks side by side, comparing their sizes.

    Parameters:
    ----------
    A1 : array-like
        The first network adjacency matrix.

    A2 : array-like
        The second network adjacency matrix.

    real_axis : int, optional
        The axis along which the networks are considered real (default is 1).
        If real_axis=1, the rows represent the networks and columns represent nodes.
        If real_axis=0, the columns represent the networks and rows represent nodes.

    ax : matplotlib Axes, optional
        The axes on which to draw the heatmap (default is None).
        If None, a new figure and axes will be created.

    linewidth : float or None, optional
        The width of the lines separating cells in the heatmap (default is None).

    linecolor : color or None, optional
        The color of the lines separating cells in the heatmap (default is None).

    cmap : str or colormap, optional
        The colormap to be used for the heatmap (default is None).

    cbar : bool or None, optional
        Whether to draw a colorbar or not (default is None).
        If None, a colorbar is drawn if cmap is specified.

    Returns:
    -------
    None

    Examples:
    ---------
    >>> A1 = np.random.rand(10, 10)
    >>> A2 = np.random.rand(10, 10)
    >>> plot_nets_side_by_size(A1, A2)

    >>> plot_nets_side_by_size(A1, A2, real_axis=0, cmap='coolwarm')
    """
    # real axis is the num of the nets
    if checkEmptyList(ax): fig, ax = plt.subplots()
    conc = []
    if real_axis == 1:

        A1_A2 = np.hstack([np.hstack([ norm_vec_min_max(A1[:,p]).reshape((-1,1)), norm_vec_min_max(A2[:,p]).reshape((-1,1))])  for p in range(A1.shape[1])])
    else:
        A1_A2 = np.vstack([np.vstack([ norm_vec_min_max(A1[p,:]).reshape((1,-1)), norm_vec_min_max(A2[p,:]).reshape((1,-1))])  for p in range(A1.shape[0])])
        
    sns.heatmap(A1_A2, ax = ax, robust = True, linewidth = linewidth, linecolor = linecolor,cmap =cmap , cbar = cbar )
        
        
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
        print('loaded data is')
        print(data_type + '_%d'%params['addi_name'])
      
        return data_type, create_data_name(data_type + '_%d'%params['addi_name'], xmin, xmax, ymin, ymax, type_name)
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
    """
    Split stacked data into separate blocks.

    Parameters:
    ----------
    data : array-like
        The stacked data in the shape Neurons x (Time x Conditions).
        If the data has shape (Neurons x Time x Conditions), each condition is split separately.

    T : int, optional
        The length of each block in the time dimension (default is 0).
        If T is 0, it will be inferred from the data shape.

    k : int, optional
        The number of blocks in the condition dimension (default is 0).
        If k is 0, it will be inferred from the data shape.

    Returns:
    -------
    array-like
        The split data in the shape Neurons x Time x Conditions.

    Raises:
    ------
    ValueError
        If both T and k are 0 or if T * k is not equal to the second dimension of the data.

    Examples:
    ---------
    >>> data = np.random.rand(100, 500)  # Neurons x (Time x Conditions)
    >>> split_data = split_stacked_data(data, T=100, k=5)  # Neurons x Time x Conditions

    >>> data = np.random.rand(100, 500, 2)  # Neurons x Time x Conditions
    >>> split_data = split_stacked_data(data)  # Neurons x Time x Conditions
    """    
    # data in shape Neurons X (Time X conditions) [such that [bluck 1 N X T] , bluck 2 N X T]
    # PAY ATTNETION !! RETURN N x TIME x CONDITION (if applying for phi in grannet - transpose axis 0,1)
    if len(data.shape) == 3:
        return np.dstack([split_stacked_data(data[:,:,k_global], T, k) for k_global in range(data.shape[2])] )
    if T == 0 and k == 0:
        raise ValueError('you must provide either k and T!')
    elif T != 0 and k!= 0:
        if k*T != data.shape[1]:
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
    store_data = []
    for opt_count, opt_begin in enumerate(groups_opts[:-1]):
        opt_end = groups_opts[opt_count + 1]
        cur_data = data[:,opt_begin : opt_end ]
        store_data.append(cur_data)
    return np.dstack(store_data)
        
        
def  create_proximity_matching(data, labels,  increase_diag = True) :
    """
    Create a proximity matching matrix based on the given data and labels.

    Parameters:
    ----------
    data : array-like
        The original 3D matrix of shape Neurons x Time x Trials.

    labels : array-like
        The labels associated with each trial.

    increase_diag : bool, optional
        Whether to increase the values on the diagonal of the proximity matrix (default is True).

    Returns:
    -------
    array-like
        The proximity matching matrix of shape n x n, where n is the number of unique labels.

    Raises:
    ------
    ValueError
        If the duration of the data and labels are different.

    Examples:
    ---------
    >>> data = np.random.rand(100, 500, 100)  # Neurons x Time x Trials
    >>> labels = np.random.randint(0, 5, 100)  # Labels per trial
    >>> proximity_mat = create_proximity_matching(data, labels)
    """    
    # the data is the original 3d mat
    # labels here mmust be labels full (labels per trial)
    
    un_labels = np.unique(labels)
    sim_mat =  np.zeros(( len(un_labels),len(un_labels), data.shape[0])) 
    if len(labels) != data.shape[2]:
        raise ValueError('data nd labels have different durations! please provide full labels info')
   
    for neuron in range(data.shape[0]):
        list_datas = []
        # create matrices of trials X time X neuron for each state
        
        for label in un_labels :
            data_label = data[:,:,labels == label]
            data_label_stack = np.transpose(data_label, (2,1,0))
            list_datas.append(data_label_stack[:,:,neuron]) # FUTURE IMPROVEMENT - MAKE IT MUCH MORE EFFICIENTS W/O RUNNING ON NEURONS 
    
        # now I have a list of all states in the form of trials X time for each neuron

        pairs = list(itertools.combinations(range(len( list_datas) ), 2))
        for pair in pairs:
            """
            try to solve psi Y_1 - Y_2
            """
            Y_1 = list_datas[pair[0]]
            Y_2 =  list_datas[pair[1]]
            Y_2_Y_1_T = Y_2 @ Y_1.T
            U, s, Vh = np.linalg.svd(Y_2_Y_1_T, full_matrices=False)


            psi = U @ Vh

            Y1_trans = psi @ Y_1
            dist = np.sqrt(np.mean((Y1_trans - Y_2)**2))
            sim_mat[pair[0], pair[1], neuron] = dist
            sim_mat[pair[1], pair[0], neuron] = dist
    neuron_avg = np.mean(sim_mat, 2)

    # increase diag
    if increase_diag: 
        neuron_avg = neuron_avg +  np.eye(neuron_avg.shape[0])*10**3
       
        
        neuron_avg += np.eye(neuron_avg.shape[0])*np.min(neuron_avg)
    return neuron_avg
            
            
            
    
    
   
    
    
    
def run_GraFT(data = [], corr_kern = [], params = {}, to_save = True, to_return = True,
               ask_selected = ask_selected, selected = ['epsilon','step_s', 'p', 'nonneg','step_decay'
                                               ,'solver', 'l4', 'l1'], grannet = False,
               label_different_trials = [], save_mid_results = True, type_answer = type_answer,
               instruct_per_select = instruct_per_selected, nu = [], images = False, data_type = '', path_name = '',
               labels_name = '', labels = []):
    """
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! This is the main function for the SiBBlInGS algorithm !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    Parameters
    ----------
    data : can be string of data path or a numpy array (pixels X pixels X time) or (pixels X time).
        Leave empty for default.
        The default is []. In this case, the calcium imaging dataset will be used. If grannet is True,
        data should be a 3D array of shape (pixels X pixels X time).

    corr_kern : proximity kernel. Leave empty ([]) to re-create the kernel.

    params : dictionary of parameters, optional.
        The full default values of the optional parameters are mentioned in dict_default.

    to_save : boolean, optional.
        Whether to save the results to .npy file. The default is True.

    to_return : boolean, optional.
        Whether to return results. The default is True.

    ask_selected : boolean, optional.
        Whether to ask the user about specific parameters. The default is True.

    selected : list of strings, optional.
        Relevant only if 'ask_selected' is True.
        The default is ['epsilon', 'step_s', 'p', 'nonneg', 'step_decay', 'solver', 'l4', 'l1'].

    grannet : boolean, optional.
        Whether to use the GraNNET/SiBBlInGS algorithm. If True, it means that there is more than 1 state, and the data should be a 3D array (channels X time X states).

    label_different_trials : list, optional.
        Labels for different trials. Required only for supervised or shared_flex GraNNET.
        The default is an empty list.

    save_mid_results : boolean, optional.
        Whether to save intermediate results. The default is True.

    type_answer : function, optional.
        A function to get user input. The default is type_answer.

    instruct_per_select : dict, optional.
        A dictionary containing instructions for each parameter in 'selected'. The default is instruct_per_selected.

    nu : list or np.array, optional.
        Describing how much weight to give to different nets. Its length must be equal to 'p' or empty.

    images : boolean, optional.
        If True, assume data is a sequence of images (pixels X pixels X time). The default is False.

    data_type : str, optional.
        Type of the data. The default is an empty string.

    path_name : str, optional.
        Path to save the results. If empty, it is determined based on 'data_type' and current date.

    labels_name : str, optional.
        Name of the labels file. Required only if labels are not provided. The default is an empty string.

    labels : list, optional.
        Labels for the data. Required only if 'label_different_trials' is not provided. The default is an empty list.

    **kwargs : additional keyword arguments.
        Additional arguments to be passed.

    Raises
    ------
    ValueError
        If invalid path.

    Returns
    -------
    A : np.ndarray (pixels X p)
        Neural maps.

    phi : np.ndarray (time X p)
        Temporal traces.

    additional_return : dict
        Dictionary with additional returns, including error over iterations.

    A : np.ndarray (pixels X p) - neural maps / BBs
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
    
    if to_save:
        save_name =  str(datetime2.now()).replace(':','_').replace('.','_').replace('-','_')+'_' + str(np.random.randint(0,10)) + '_g'
       
        if ask_selected:
            addition_name = '_'.join([s +'-' + str(params[s] ) for s in selected])
            save_name_short = save_name
            save_name = save_name + '_' + addition_name
        print(save_name)
     
    else:
        save_name = 'def'
  
 
    """
    create path name
    """
    if len(path_name) ==0:
        path_name = data_type + os.sep + str(date.today()) + os.sep 
    """
    type name labels
    """
    type_name_labels = 'labels_full'
    
    """
    Createdata  
    """
    default = False
    if checkEmptyList(data):
               
        default = True
        
        data_type, data = load_data(data_type = [])
       
        if checkEmptyList(label_different_trials) and grannet:
            try:
                _, label_different_trials = load_data(data_type = data_type, type_name=type_name_labels, **params[data_type] )
            except:
                _, label_different_trials = load_data(data_type = data_type, type_name='labels', **params[data_type] )
    elif isinstance(data, str) and len(data_type) == 0:
        if data.endswith('.npy'):
            data_type = data.split('_')[1]
        else:
            data_type = data
    """
    label_different_trials for supervised
    """
    if isinstance(data, str): # Check if path
        been_cut = False
        if grannet:
            data = np.load(data)  
            been_cut = True
            if checkEmptyList(label_different_trials):
                try:
                    name_label = create_data_name(data_name = data_type,  type_name = type_name_labels, **params[data_type]) 
                except:
                    name_label = create_data_name(data_name = data_type,  type_name = 'labels', **params[data_type]) 

                label_different_trials =  np.load(name_label)
                
                
            
        else:
            try:
                try:
                    if data.endswith('.npy'):
                         
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
            if np.max(labels.shape) == len(labels.flatten()):
                lab_unique, trials_counts = np.unique(labels, return_counts = True)     
            else:

                lab_unique, trials_counts = np.unique(labels, return_counts = True, axis = 1)     
            
        except:
            raise ValueError('unable to load labels, try %s'%labels_name)        
        params['trials_counts'] = trials_counts
        params['lab_unique'] = lab_unique
        if params['is_trials_type'] == 'shared_flex': # same kernel, different nets
            # future change kernel
            data_full = data.copy()            
            # make data for each label
            # each el will be neurons X Time X trials for k_spec
            data =np.dstack( [MovToD2_grannet(data[:,:,labels == k]) for  k in lab_unique])
            
        elif params['is_trials_type'] == 'flex': 
            pass
        elif params['is_trials_type'] == 'shared':  # same kernel same nets
            """
            in this case - the kernel witl be only k d (no trials included) and A will also be only k d. however
            Y and phi will be kXtirals dimensions
            """
            data_full = data.copy()            
            # make data for each label
            # each el will be neurons X Time X trials for k_spec
            data = [MovToD2_grannet(data[:,:,labels == k]) for  k in lab_unique]
            try:
                data = np.dstack(data)
            except:
                print('data hve different trials')
            
            
        else:
            raise ValueError("how you arrive here? params_default['is_trials_type'] %s"%params['is_trials_type'])
    
    
    
    
    """
    define save path
    """
    if to_save:
        path_name = data_type + os.sep + str(date.today()) 
        path_name = path_name ='results_march_2023' + os.sep + data_type +  os.sep +str(date.today()) + os.sep +  params['name_addition'] + save_name + '_folder' 
        #

        
    if images:
        if params['drop_med']:
            if  params['likely_from'] == 'gaussian' :
                print('reduce min not median since poisson')
                data = np.dstack([data[:,:,k] - np.median(data[:,:,k]) for k in range(data.shape[2])])
            else:
                data = data - np.min(data)
              
                
        params['base_shape'] = data[:,:,0].shape
        data = MovToD2(np.dstack(data))
     
        
    if isinstance(data,np.ndarray) and ((is_2d(data, 3) and not grannet) or (is_2d(data,2) and grannet)):
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
        if isinstance(data,np.ndarray):
            data = (data - np.min(data, 1).reshape((-1,1))) / (data.max(1) - data.min(1)).reshape((-1,1))
        else:
            print('different trials')
    
    """
    create grannet labels 
    """

    if grannet:
        """
        unseupervised
        """
        if params['condition_unsupervised']:
            if isinstance(data, np.ndarray):
                num_states = data.shape[2]
            else:
                num_states = len(data)
            proximity = []
        

        else:
            """
            seupervised
            """
            """
            this is the case where we HAVE labels and we know what to expect. For instance disase stage 1,2,3,4. We do not want to use the data itself!!
            """
            if params['is_trials_type'] == 'shared':
                if is_1d(label_different_trials):
                    label_different_trials_prob = np.unique(label_different_trials)
                else:
                    label_different_trials_prob = np.unique(label_different_trials, axis = 1)

            else:
                label_different_trials_prob = label_different_trials
            _, proximity = labels2proximity(label_different_trials_prob, 
                                            distance_metric = params['grannet_params']['distance_metric'], 
                                            distance2proximity_trans = params['grannet_params']['distance2proximity_trans'], 
                                            rounded =  params['grannet_params']['rounded'],
                                           rounded_max =  params['grannet_params']['rounded_max'], params = params
                                            )
            

            num_states = proximity.shape[0]
            

  
        if params['is_trials'] and params['is_trials_type'] == 'shared' and params['condition_unsupervised']:
            """
            check if multi trial
            """
            # IF multi trial
            if np.min(np.unique(label_different_trials, return_counts=True)[1]) > 1  or len(np.unique(label_different_trials)) != len(np.unique(labels)):
                data_cur = data_full.copy()
               
                
                proximity = create_proximity_matching(data_cur, label_different_trials, True) # data cur is the original 3d mat
            
            else:
                proximity = []
            
                if isinstance(data, np.ndarray) and len(np.unique(label_different_trials)) == len(np.unique(labels)):
                    data = np.dstack([MovToD2_grannet(data[:,:,labels == k]) for  k in lab_unique])
                elif  len(np.unique(label_different_trials)) == len(np.unique(labels)):
                    data = np.dstack(data)
                    data = np.dstack([MovToD2_grannet(data[:,:,labels == k]) for  k in lab_unique])

                else:
                    raise ValueError('cannot be possible. data.shape is % and unique label len is %s'%(str(data.shape), len(np.unique(labels))))
                
            coefficients_similar_nets = create_proximity_coeffs_based_on_prxoimity(params['p'], proximity,
                                                                                   num_free_nets = params['grannet_params']['num_free_nets'], 
                                                                                   reg_different_nets = params['grannet_params']['reg_different_nets'],
                                                                                   num_states = len(np.unique(labels)), nu = nu, params = params, 
                                                                                   data = data) 

      
        elif params['condition_unsupervised']:
            coefficients_similar_nets = create_proximity_coeffs_based_on_prxoimity(params['p'], proximity, num_free_nets = params['grannet_params']['num_free_nets'], 
                                                                                   reg_different_nets = params['grannet_params']['reg_different_nets'],
                                                                                   num_states = num_states, nu = nu, params = params, data =data) 
        else:

            coefficients_similar_nets = create_proximity_coeffs_based_on_prxoimity(params['p'], proximity, num_free_nets = params['grannet_params']['num_free_nets'], 
                                                                                   reg_different_nets = params['grannet_params']['reg_different_nets'],
                                                                                   num_states = num_states, nu = nu, params = params, data =[])     


        
        """
        here increase coeffs
        """

        coefficients_similar_nets *= params['weight_sim_nets'] # HERE MUPLTIBPLY COEFFS

    
            



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
              
                
                shared_kernel = mkDataGraph( MovToD2_grannet(data), params, reduceDim = params['reduceDim'], 
                                     reduceDimParams = {'alg':'PCA'}, graph_function = 'gaussian',
                            K_sym  = True, use_former = use_former, data_name = data_type,
                            toNormRows = True,  graph_params = params['graph_params'], grannet = grannet)
                corr_ker = np.dstack([shared_kernel]*data.shape[2])

            elif params['graph_params']['kernel_grannet_type'] in ['combination',  'averaged']:
                # THIS OPTION USE A COMBINATION OF SHARED KERNEL AND INDIVIDUAL KERNEL
          
                corr_kern = mkDataGraph_grannet(data, params, reduceDim = params['reduceDim'], 
                                     reduceDimParams = {'alg':'PCA'}, graph_function = 'gaussian',
                            K_sym  = True, use_former = use_former, data_name = data_type,
                            toNormRows = True,  graph_params = params['graph_params'], 
                            grannet = grannet)
            else:
                raise ValueError(" params['graph_params']['kernel_grannet_type'] is incorrect")

         
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
         
                if not os.path.exists(path_name):
                    os.makedirs(path_name)
               
                if len(path_name) > 0:
                    np.save(path_name + os.sep+ 'kernel_%s_xmin_%s_xmax_%s_ymin_%s_ymax_%s.npy'%(data_type, str(params[data_type]['xmin']),
                                                                              str(params[data_type]['xmax']),
                                                                              str(params[data_type]['ymin']),
                                                                              str(params[data_type]['ymax'])), corr_kern)
                    
                else:
                    np.save('kernel_%s_xmin_%s_xmax_%s_ymin_%s_ymax_%s.npy'%(data_type, str(params[data_type]['xmin']),
                                                                              str(params[data_type]['xmax']),
                                                                              str(params[data_type]['ymin']),
                                                                              str(params[data_type]['ymax'])), corr_kern)
                   
                
                print('kernel saved in %s!'%(path_name + 'kernel_%s_xmin_%s_xmax_%s_ymin_%s_ymax_%s.npy'))
                
                
                
                
        elif isinstance(corr_kern, str): # Check if path
          try:
              if len(path_name) > 0:
                  try:
                      corr_kern = np.load(path_name + 'kernel_%s_xmin_%d_xmax_%d_ymin_%d_ymax_%d.npy'%corr_kern)
                  except:
                      corr_kern = np.load(path_name + 'kernel_%s_xmin_%d_xmax_%d_ymin_%d_ymax_%d.npy'%(data_type, params[data_type]['xmin'],params[data_type]['xmax'],params[data_type]['ymin'],params[data_type]['ymax']))
                      
              else:
                  try:
                      corr_kern = np.load('kernel_%s_xmin_%d_xmax_%d_ymin_%d_ymax_%d.npy'%corr_kern)
                  except:
                      corr_kern = np.load('kernel_%s_xmin_%d_xmax_%d_ymin_%d_ymax_%d.npy'%(data_type, params[data_type]['xmin'],params[data_type]['xmax'],params[data_type]['ymin'],params[data_type]['ymax']))
                      
          except:
            
              raise ValueError('Cannot locate kernel path! (your invalid path is %s)'%corr_kern)
        
        if params['is_trials'] and params['is_trials_type'] == 'shared':
            
            if coefficients_similar_nets.shape[0] != len(np.unique(labels)):
                # in this case I except coefficients_similar_nets to be unique(k) X  unique(k) X nets
                raise ValueError('Dimensions mismatch. Data shape is '+ str(len(np.unique(labels))) + ' while coeffs shape is ' + str(coefficients_similar_nets.shape))

        elif data.shape[2] != coefficients_similar_nets.shape[0]:
            """
            if kernel is shared and nets are shared then __. 
            """

        
            raise ValueError('Dimensions mismatch. Data shape is '+ str(data.shape) + ' while coeffs shape is ' + str(coefficients_similar_nets.shape))

    else: 
        """
        graft case (not grannet)
        """


        if checkEmptyList(corr_kern):
           
            print('IT IS EMPTY LIST!')
      
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
                

            
            
        elif isinstance(corr_kern, str): # Check if path
          try:
              print('kernel_%s_xmin_%d_xmax_%d_ymin_%d_ymax_%d.npy'%(data_type,params[data_type]['xmin'],params[data_type]['xmax'],params[data_type]['ymin'],params[data_type]['ymax']))
              if len(path_name) > 0:
                  corr_kern = np.load(path_name + 'kernel_%s_xmin_%d_xmax_%d_ymin_%d_ymax_%d.npy'%(data_type,params[data_type]['xmin'],params[data_type]['xmax'],params[data_type]['ymin'],params[data_type]['ymax']))

              else:
                  corr_kern = np.load('kernel_%s_xmin_%d_xmax_%d_ymin_%d_ymax_%d.npy'%(data_type,params[data_type]['xmin'],params[data_type]['xmax'],params[data_type]['ymin'],params[data_type]['ymax']))

          except:
              raise ValueError('Cannot locate kernel path! (your invalid path is %s)'%corr_kern)
        else:
            raise ValueError('Kernel should be an empty list or str. Currently it is a ' + str(type(corr_kern)))

    """
    data prep: remove median / data pre proc
    """        
    if params['drop_med']:
        if grannet:
            data = np.dstack([data[:,:,k] - np.median(data[:,:,k]) for k in range(data.shape[2])])
        elif not images:
            data = data - np.median(data)
  
    
    
    
    
    """
    save params
    """
    if to_save:
      
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
        
    
    if to_save:
       

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
    
        if grannet:
            return full_A, full_phi, additional_return
        return A, phi, additional_return

def order_A_results(full_A, full_phi):
    """
    Reorders the neural maps and their corresponding temporal traces based on the order of the first trial.
    PLOTTING A
    order A2 according to A1

    Parameters
    ----------
    full_A : np.ndarray (pixels X p X trials)
        Neural maps for all trials.

    full_phi : np.ndarray (time X p X trials)
        Temporal traces for all trials.

    Returns
    -------
    full_A_new : np.ndarray (pixels X p X trials)
        Reordered neural maps.

    full_phi_new : np.ndarray (time X p X trials)
        Reordered temporal traces.
    """    

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
    """
    Normalize a 2D matrix column-wise for plotting purposes.

    Parameters
    ----------
    mat_2d : np.ndarray
        Input matrix of shape (m, n).

    epsilon : float, optional
        Small constant added to the denominator to avoid division by zero.

    Returns
    -------
    np.ndarray
        Normalized matrix of shape (m, n).
    """    
    return np.hstack([norm_vec_min_max(mat_2d[:,t]).reshape((-1,1)) for t in range(mat_2d.shape[1])]) 


def norm_vec_min_max(vec)  :
    """
    Normalize a 1D vector using the minimum and maximum values.

    Parameters
    ----------
    vec : np.ndarray
        Input vector.

    Returns
    -------
    np.ndarray
        Normalized vector.
    """    
    return (vec - np.min(vec))/(vec.max() - np.min(vec))
    
          
def GraFT_with_GraNNet(data, phi, kernel, params,  coefficients_similar_nets = [], grannet = True,  seed = 0,
                       to_store_lambdas = params_default['to_store_lambdas'], 
                       save_mid_results = True, path_save = '', T_save = 10, dataname = 'unknown', 
                       save_name = 'def', path_name = '', labels = []):
    """
    Function to learn a dictionary for spatially ordered/graph-based data using a re-weighted l1 spatial/graph filtering model.

    Parameters
    ----------
    data : np.array
        neurons X time OR (for grannet:) neurons X time X trials
    phi : TYPE
        (time, p)
    kernel : neurons X neurons
        DESCRIPTION.
    params : TYPE
        DESCRIPTION.
    coefficients_similar_nets : list, optional
        For further applications (GraNNet). Default is an empty list.
    grannet : bool, optional
        Flag indicating whether to use GraNNet. Default is True.
    seed : int, optional
        Random seed. Default is 0.
    to_store_lambdas : bool, optional
        Flag indicating whether to store lambdas. Default is the value of `params_default['to_store_lambdas']`.
    save_mid_results : bool, optional
        Flag indicating whether to save intermediate results. Default is True.
    path_save : str, optional
        Path to save intermediate results. Default is an empty string.
    T_save : int, optional
        Frequency of saving. Default is 10.
    dataname : str, optional
        Name of the data. Default is 'unknown'.
    save_name : str, optional
        Name for saving. Default is 'def'.
    path_name : str, optional
        Path name. Default is an empty string.
    labels : list, optional
        List of labels. Default is an empty list.

    Returns
    -------
    None.

    """

    params['solver_or'] = params['solver']
    np.random.seed(seed)
    if grannet and len(data.shape) != 3: 
        raise ValueError('Data Should be a 3-dim array in case of GraNNet!')
    additional_return  = {'MSE':[]}
    if len(data.shape) == 3 and not grannet: 
        data = MovToD2(data)
    params = {**{'max_learn': 1e3, 'learn_eps': 0.01,'step_decay':0.995}, **params}
   
    n_rows = data.shape[0] # number of neurons, N
    n_cols = params['p']

    n_times = data.shape[1]
    extras = {'dictEvo':[], 'presMap':[], 'wghMap':[]} # Store some outputs
    
    """
    Initialize dictionary
    """    
    if params['to_sqrt']:
        multi = np.sqrt(np.abs(np.mean(data)))
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
        initialize A !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        """
        if params['init_same']:
      
            if params['dist_init'] == 'zeros':
                full_A = np.zeros((n_rows, n_cols, 1))
                if to_store_lambdas:  full_lambdas = np.zeros((n_rows, n_cols, n_states))
            elif params['dist_init'] == 'rand':
                full_A = np.random.rand(n_rows, n_cols, 1)*multi
                if to_store_lambdas: full_lambdas = np.random.rand(n_rows, n_cols,1)*multi
            elif params['dist_init'] == 'randn':
                full_A = np.random.randn(n_rows, n_cols, 1)*multi
                if to_store_lambdas: full_lambdas = np.random.randn(n_rows, n_cols,1)*multi
            elif params['dist_init'] == 'uniform':
                full_A = np.random.uniform(0, 1,size = (n_rows, n_cols, 1))*multi
                if to_store_lambdas: full_lambdas = np.random.uniform(0, 1,size = (n_rows, n_cols,1))*multi        
                
            else:
                raise ValueError('Unknown dist init')
            full_A = np.dstack([full_A]*n_states)
            if to_store_lambdas: full_lambdas = np.dstack([ full_lambdas ]*n_states)
        else:
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
        if np.isnan(multi): input('multi is nan')

        
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

    if params['initial_thres']:
        A_d = np.dstack([hard_thres_on_A(full_A[:,:,k], params['hard_thres_params']['non_zeros']) for k in range(full_A.shape[2])])

        full_A[:,:-1,:] = A_d[:,:-1,:]
    if params['norm_A_cols'] and grannet:
        full_A  = full_A / np.expand_dims(np.sum(np.abs(full_A), 1),1)*10
        
    if params['norm_A']:
        # NOEMALIZE EACH COL OF A TO MIN MAX
        if params['norm_A_params']['type_norm_A'] == 'min_max':
            if grannet: 
                full_A  = full_A / np.expand_dims(np.sum(np.abs(full_A), 0),0)*10
            else:
                A = A/np.sum(np.abs(A),0)*10
        else:
            raise ValueError('not available yet!')
        

        
        
        
    if params['num_nets_sample_trials'] != 0  :
        full_A_original = full_A.copy()
        full_phi_original = full_phi.copy()
        raise ValueError('not fully implemented yet! need to do the actualy choosing later - future work futuresteps')
                            
    fig, axs = plt.subplots(2, full_A.shape[2], figsize = (20,10)) 
    vmin = np.percentile(full_A.flatten(), 5)
    vmax = np.percentile(full_A.flatten(), 95)
    vmin = - np.max([np.abs(vmin), np.max(vmax)])
    vmax = -vmin
    [sns.heatmap(full_A[:,:,k], ax = axs[0,k], robust = True, vmin = vmin, vmax = vmax, cmap = 'bwr') for k in range(full_A.shape[2])] 
    [sns.heatmap(full_phi[:,:,k*int(full_phi.shape[2]/full_A.shape[2])], ax = axs[1,k], robust = True) for k in range(full_A.shape[2])]
    [ax_s.set_xticks([]) for ax_s in axs.flatten()]
    [ax_s.set_yticks([]) for ax_s in axs.flatten()]
    fig.tight_layout()
    name_save = 'iter_%d_error100_%s'%(n_iter, str(cur_error*100).replace('.','dot'))

    plt.savefig(path_save +  name_save + '.png')
        
    plt.close()
        
    data_or_full = data.copy()
    phi_or_full = full_phi.copy()
    labels_full = labels.copy()

    if grannet and params['is_trials'] and params['boots_trials']:
        boots_size = np.min([params['boots_size'], np.min(params['trials_counts'])])

    if np.isnan(full_A).any():

        full_A[np.isnan(full_A)] = 0
        full_A += np.expand_dims(np.random.randn(*full_A.shape[:2])*0.05, 2)

        
        
    """
    nullify first connection
    """
    if params['null_first']:
        coefficients_similar_nets[0,:] = 0
        print('pay attention, you nullify first state connection1!!!!!!!!!!!!!')
        

        
    """
    'norm_by_sum'
    """
    if params['norm_by_sum']:
        data = data*np.expand_dims(data.max(1),1)*10 / (np.expand_dims(data.sum(1), 1))

        
        
        
    while n_iter < params['max_learn'] and (error_dict > params['dict_max_error'] or cur_error > params['mean_square_error']):
        """
        decay similariy between nets
        """
        if  params['graph_params']['increase_sim'] != 1  and grannet:
            coefficients_similar_nets *=  params['graph_params']['increase_sim']
        
        
        if params['solver'].lower() == 'spgl1':
            params['l1'] *= params['params_spgl1']['decay']
            
        if grannet and params['is_trials'] and (params['boots_trials'] or not (params['trials_counts'] == params['trials_counts'][0]).all()):
            """
            RESTORE ORIGINAL DATA
            """
            if n_iter > 0:
                data = data_or_full.copy()
                labels = labels_full.copy()
                phi_or_full[:,:,trial_selected] = full_phi
            """
            choose selected trials
            """
           
            locs_choose = []
            for lab_count, lab in enumerate(np.unique(labels)): # k in range(len(params['trials_counts'])
                np.random.seed(n_iter + lab_count * n_iter)
                locs = np.where(labels == lab)[0]
                locs_choose.append(list(np.random.choice(locs , replace = False, size = boots_size)))
            trial_selected = np.array(locs_choose).flatten()
            
            """
            compress data and phi
            """
            data = data[:,:,trial_selected]
            labels = labels_full[trial_selected]
            full_phi = phi_or_full[:,:,trial_selected]

            params['trials_counts_update'] = boots_size*np.ones(len(np.unique(labels)))
            n_states_phi = full_phi.shape[2]
        elif params['is_trials'] and params['is_trials_type'] == 'shared': 
            params['trials_counts_update'] = params['trials_counts'].copy()
        if params['is_trials'] and params['is_trials_type'] == 'shared': 
            params['trials_counts_update'] =  params['trials_counts_update'].astype(int)

        n_iter += 1

        if step_GD < params['min_step_size']:
            step_GD = params['min_step_size']
        if step_GD > params['max_step_size']:
            step_GD = params['max_step_size']

        
        """
        CHOOSE NETS and stack trials
        """
        if params['num_nets_sample_trials'] != 0:
            if not params['is_trials'] :
                raise ValueError('only for trials case! please set is_trials to True')
            

            raise ValueError('not implemented yet!')
            
        if params['is_trials'] and params['is_trials_type'] == 'shared': # before the update of A
            full_A_original = full_A.copy()
            full_phi_original = full_phi.copy()           
            data_original = data.copy()

            full_phi = np.dstack([MovToD2_grannet(full_phi[:,:,labels == k].transpose((1,0,2))).T                                 
                                  for  k in params['lab_unique']])

        
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
                if params['one_inv'] and np.mod(n_iter, params['T_inv']) == 0:
                    params['solver'] = 'inv'
                elif  np.mod(n_iter, params['T_inv']) != 0:
                    params['solver'] = params['solver_or']


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
            if params['one_inv'] and np.mod(n_iter, params['T_inv']) == 0:
                params['solver'] = 'inv'
            elif  np.mod(n_iter, params['T_inv']) != 0:
                params['solver'] = params['solver_or']
            A, lambdas = dictionaryRWL1SF(data, phi, kernel, params = params, A=A) # Infer coefficients given the data and dictionary

            if np.abs(A.mean()) < 1e-39:
                A = A + init_mat(shape = A.shape, dist_init = 'normal', multi = np.median(data))
                print('A mean after noise normal')
                print(str(A.mean()))
                

        if np.isnan(full_A).any():

            full_A[np.isnan(full_A)] = 0
            full_A += np.expand_dims(np.random.randn(*full_A.shape[:2])*0.05, 2)
 
        """
        take hard thres on A
        """
        if params['hard_thres'] and cur_error < params['hard_thres_params']['thres_error_hard_thres'] and np.mod(n_iter, params['hard_thres_params']['T_hard']) == 0:
            if grannet: 
                if params['hard_thres_params']['keep_last']:
                    A_d = np.dstack([hard_thres_on_A(full_A[:,:,k], params['hard_thres_params']['non_zeros']) for k in range(full_A.shape[2])])
                    #full_A = #np.dstack([A_d, full_A[:,:,-1]])
                    full_A[:,:-1,:] = A_d[:,:-1,:]
                else:
                    full_A = np.dstack([hard_thres_on_A(full_A[:,:,k], params['hard_thres_params']['non_zeros']) for k in range(full_A.shape[2])])

            else:
                if params['hard_thres_params']['keep_last']:
                    A_d = hard_thres_on_A(A, params['hard_thres_params']['non_zeros']) 
                
                    A[:,:-1] = A_d[:,:-1]
                else:
                    A = hard_thres_on_A(A, params['hard_thres_params']['non_zeros']) 
        
            print('APPLIED HARD THRES!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            
        """
        fill nans in A
        """

        if np.isnan(full_A).any():

            full_A[np.isnan(full_A)] = 0
            full_A += np.expand_dims(np.random.randn(*full_A.shape[:2])*0.05, 2)

        """
        make sure A is not crazy
        """
        if grannet: 
            if np.percentile(full_A, 95) > 10**8:
                full_A = full_A/np.median(full_A)
                if np.max(full_A) > 10**8:
                    full_A = full_A/np.max(full_A)
        else:
            if np.max(A) > 10**8:
                A = A/np.median(A)
                if np.max(A) > 10**8:
                    A = A/np.max(A)   
                    
        """
        normalize A if parameters asks
        """
   
        if params['norm_A']:
            # NOEMALIZE EACH COL OF A TO MIN MAX
            if params['norm_A_params']['type_norm_A'] == 'min_max':
                if grannet: 
                    full_A  = full_A / np.expand_dims(np.sum(np.abs(full_A), 0),0)*10
                    full_A[np.isnan(full_A)] = 0
                else:
                    A = A/np.sum(np.abs(A),0)*10
            else:
                raise ValueError('not available yet!')
        """
        fill nans in A after normalizing
        """
        if np.isnan(full_A).any():

            full_A[np.isnan(full_A)] = 0
            full_A += np.expand_dims(np.random.randn(*full_A.shape[:2])*0.05, 2)

                        
                
        if grannet: 
            
            if params['is_trials'] and params['is_trials_type'] == 'shared': # for updating phi

                
                if (params['trials_counts_update'] == params['trials_counts_update'][0]).all():
                    full_phi = split_stacked_data(full_phi.transpose((1,0,2)), T = full_phi_original.shape[0],
                                                  k = params['trials_counts_update'][0] ).transpose((1,0,2)) 

                else:
                    raise ValueError('fsomething is wrong. should be')
                data = data_original.copy()
                
                full_A_original = full_A.copy()
                full_A = np.repeat(full_A, params['trials_counts_update'], axis = 2)
                
                            
            """
            fill nans in A
            """
            if np.isnan(full_A).any():

                full_A[np.isnan(full_A)] = 0
                full_A += np.expand_dims(np.random.randn(*full_A.shape[:2])*0.05, 2)

                            
                    
                
            """
            update phi !
            """
            dict_old = full_phi[:,:,trial_num] # Save the old dictionary for metric calculations
            """
            find phi by inverse
            """
            
            if params['add_inverse'] and np.mod(n_iter, params['inverse_params']['T_inverse']) == 0:
                for trial_num in range(n_states_phi):
                    if not  params['phi_positive']:
                        phi = data[:,:,trial_num].T @ np.linalg.pinv(full_A[:,:,trial_num]).T 
                    else:
                        print(full_phi.shape)
                        print('full phi shape')
                        phi = np.hstack([(nnls(full_A[:,:,trial_num],  data[:,t,trial_num].flatten())[0]).reshape((-1,1))
                                         for t in range(full_phi.shape[0])]).T
                        print(phi.shape)
                      
                    
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

                        phi_selected = phi_list[argmin]
                        if cur_error_list[argmin] < cur_error_before + cur_error_before*params['min_add_phi_error'] or not params['phi_only_dec']:
                            full_phi[:,:,trial_num] = phi_selected
                        else:
                            print('did not change phi since error increased')
                        step_selected_list.append(steps_opts[argmin]  )
            
                    step_GD   = np.median(step_selected_list )   

                    step_GD   = step_GD*params['step_decay']     
                else:
                    for trial_num in range(n_states_phi):
                        phi = dictionary_update(full_phi[:,:,trial_num], full_A[:,:,trial_num], data[:,:,trial_num], 
                                                step_GD, GD_type = params['GD_type'], params = params) # Take a gradient step with respect to the dictionary     
                        full_phi[:,:,trial_num] = phi
                    step_GD   = step_GD*params['step_decay']                                                     # Update the step size 
            
            if params['add_phi_smooth']        :
                full_phi = mov_avg(full_phi, axis = 0, wind = params['wind'])
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
                

            
            """
            save results for grannet
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
                print(name_save)
                np.save( path_save +  name_save  + '.npy', 
                        {'params':params,'additional_return': additional_return, 'full_phi': full_phi,
                         'full_A':full_A, 'data':data, 'save_name':save_name, 'cur_error':cur_error})
                if 'trends' in path_save:
                    terms_and_labels_trend = np.load('grannet_trends_for_jupyter_results_march_2023.npy', allow_pickle=True).item()

                    
                    dd,_, _ = create_dict_of_clusters_multi_d_A(full_A,
                                                                labels = terms_and_labels_trend['labels'],
                                                                          terms = np.array(list(terms_and_labels_trend['terms'])))
                    
                    save_dict_to_txt(dd, path_save = path_save +  name_save  + '.txt')
                    
                    full_A_mov = MovToD2_grannet(full_A)
                    full_A_mov = full_A_mov/full_A_mov.sum(1).reshape((-1,1))
                    full_A_st = split_stacked_data(full_A_mov , k = full_A.shape[2])
                    dd,_, _ = create_dict_of_clusters_multi_d_A(full_A_st,
                                                                labels = terms_and_labels_trend['labels'],
                                                                          terms = np.array(list(terms_and_labels_trend['terms'])))
                    
                    save_dict_to_txt(dd, path_save = path_save +  name_save  + '2.txt')                    
                    
                    
                
            if params['save_plots'] and (np.mod(n_iter, T_save) == 0 and np.mod(n_iter, params['T_save_fig']) == 0 ) or n_iter  <= 3:
  
                
                fig, axs = plt.subplots(2, full_A.shape[2], figsize = (20,10)) 
                vmin = np.percentile(full_A.flatten(), 5)
                vmax = np.percentile(full_A.flatten(), 95)
                vmin = - np.max([np.abs(vmin), np.max(vmax)])
                vmax = -vmin
                [sns.heatmap(full_A[:,:,k], ax = axs[0,k], robust = True, vmin = vmin, vmax = vmax, cmap = 'bwr') for k in range(full_A.shape[2])] 
                [sns.heatmap(full_phi[:,:,k*int(full_phi.shape[2]/full_A.shape[2])], ax = axs[1,k], robust = True) for k in range(full_A.shape[2])]
                [ax_s.set_xticks([]) for ax_s in axs.flatten()]
                [ax_s.set_yticks([]) for ax_s in axs.flatten()]
                fig.tight_layout()
                plt.savefig(path_save +  name_save + '.png')
                    
                plt.close()
           
            if params['compare_to_synth'] and np.mod(n_iter, params['T_save_fig']) == 0:
                           
                full_phi_ordered, full_A_ordered = snythetic_evaluation(full_A, full_phi, A_real, phi_real)
            
                
              
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
                

                plt.savefig(path_save +  name_save + '_compare_to_real.png')
                    
                plt.close()
        
                
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

                    
        else: # (if not grannet/SiBBlInGS -> ONE STATE ONLY)
            
            if params['add_inverse'] and np.mod(n_iter, params['inverse_params']['T_inverse']) == 0:
                phi = (np.linalg.pinv(A) @ data ).T

            else:    
                if params['various_steps']:
                    steps_opts = step_GD*params['steps_range']
                    step_selected_list = []
                    
                    cur_error_list = []
                    phi_list = []

                    cur_error_before =  np.mean((A @ phi.T - data)**2)  
                    for step_opt in steps_opts:
                         
                        phi_try = dictionary_update(phi, A, data, step_opt, GD_type = params['GD_type'], params = params) # Take a gradient step with respect to the dictionary  
                        cur_error =  np.mean((A @ phi_try.T - data)**2)  
                        cur_error_list.append(cur_error)
                        phi_list.append(phi_try)
                    if params['step_loc'] == 0:
                        argmin = np.argmin(cur_error_list)
                    else:
                        sortedarg_cur_error_list = np.argsort(cur_error_list)
                        argmin = sortedarg_cur_error_list[params['step_loc']]

                    phi_selected = phi_list[argmin]
                    if cur_error_list[argmin] < cur_error_before + cur_error_before*params['min_add_phi_error'] or not params['phi_only_dec']:
                        phi = phi_selected
                    else:
                        print('did not change phi since error increased')
                    step_selected_list.append(steps_opts[argmin]  )
        
                    step_GD   = np.median(step_selected_list )   

                        
                        
                else:
                    dict_old = phi # Save the old dictionary for metric calculations
                    phi = dictionary_update(phi, A, data, step_GD, GD_type = params['GD_type'], params = params) # Take a gradient step with respect to the dictionary      
            if params['add_phi_smooth']        :
                phi = mov_avg(phi, axis = 0, wind = params['wind'])
                
            step_GD   = step_GD*params['step_decay'] 
            cur_error  = np.mean((A @ phi.T - data)**2)                                                  # store error
            
            additional_return['MSE'].append(cur_error)
            
            if step_GD < params['min_step_size']:
                step_GD = params['min_step_size']
            if step_GD > params['max_step_size']:
                step_GD = params['max_step_size']

            
            if save_mid_results and np.mod(n_iter, T_save) == 0:
                
                try:
                    path_save
                except:
                    
                    if len(path_name) ==0:
                        path_name = dataname + os.sep + str(date.today()) + os.sep 
                if len(path_save) == 0:
                    path_save =   path_name + os.sep  +  'mid_results' + os.sep
                if (not os.path.exists(path_save)) and save_mid_results and np.mod(n_iter, T_save) == 0:
                    os.makedirs(path_save)
                    print('creates path to save...')   
                name_save = 'iter_%d_error100_%s'%(n_iter, str(cur_error*100).replace('.','dot'))


                np.save( path_save + name_save  + '.npy', 
                        {'params':params, 'additional_return': additional_return, 'phi': phi, 'A':A, 'data':data,
                         'save_name':save_name, 'cur_error':cur_error})
                       
            if params['save_plots'] and  np.mod(n_iter, params['T_save_fig']) == 0 :

            
                full_A_3d = D2ToMov_inv(A, shape = params['base_shape'])
                
                fig, axs = plt.subplots(5, int(np.ceil(A.shape[1]/5)))

                axs = axs.flatten()
                [sns.heatmap(full_A_3d[:,:,j], ax = axs[j]) for j in range(A.shape[1])]
                fig.tight_layout()                
                plt.savefig(path_save +  name_save + '_A_vals_iter_%d.png'%n_iter)
                    
                plt.close()
                
                fig,ax = plt.subplots()
              
                ax.plot(phi)
                plt.savefig(path_save +  name_save + '_phi_vals_iter_%d.png'%n_iter)
                    
                plt.close()

                        
            if params['add_noise_stuck'] and len(additional_return['MSE']) > 2*params['noise_stuck_params']['in_a_row']:
                last_errors = additional_return['MSE'][-params['noise_stuck_params']['in_a_row']:]
                A, phi, step_GD = add_noise_if_stuck(last_errors, A, phi,  step_GD, params_stuck = params['noise_stuck_params'])
                params['noise_stuck_params']['std_noise_A'] *= 0.99
                params['noise_stuck_params']['std_noise_phi'] *= 0.99
                
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


def mov_avg(c, axis = 1, wind = 5):
    """
    Calculate the moving average of an input array along a specified axis.

    Parameters:
    - c: Input array
    - axis: Axis along which the moving average is calculated (default: 1)
    - wind: Size of the moving window (default: 5)

    Returns:
    - Moving average array

    Raises:
    - ValueError: If the input array shape is not supported

    """    
    if len(c.shape) == 2 and axis == 1:        
        return np.hstack([np.mean( c[:,np.max([i-wind, 1]):np.min([i+wind, c.shape[1]])],1).reshape((-1,1))
              for i in range(c.shape[1])])
    elif len(c.shape) == 2 and axis == 0:
        return mov_avg(c.T, axis = 1).T
    elif len(c.shape) == 3: # and axis == 0:
        return np.dstack([mov_avg(c[:,:,t], axis = axis) for t in range(c.shape[2])  ])
    else:
        raise ValueError('how did you arrive here? data dim is %s'%str(c.shape))
    
    


def norm(mat):
    """
    Calculate the norm of a given input matrix.

    Parameters:
    - mat: Input matrix

    Returns:
    - Norm of the matrix

    """
  
    if len(mat.flatten()) == np.max(mat.shape):
        return np.sqrt(np.sum(mat**2))
    else:
        _, s, _ = np.linalg.svd(mat, full_matrices=True)
        return np.max(s)
    
def mkCorrKern(params = {}):
    """
    Generates a correlation kernel.

    Parameters:
        params (dict, optional): Additional parameters. Default is an empty dictionary.

    Returns:
        corr_kern (object): The correlation kernel.
    """

    # Make a kernel
    params = {**{'w_space':3,'w_scale':4,'w_scale2':0.5, 'w_power':2,'w_time':0}, **params}
    dim1  = np.linspace(-params['w_scale'], params['w_scale'], 1+2*params['w_space']) # space dimension
    dim2  = np.linspace(-params['w_scale2'], params['w_scale2'], 1+2*params['time']) # time dimension
    corr_kern  = gaussian_vals(dim1, std = params['w_space'], mean = 0 , norm = True, 
                               dimensions = 2, mat2 = dim2, power = 2)
    return corr_kern
    
def checkCorrKern(data, corr_kern, param_kernel = 'embedding', recreate = False, know_empty = False):
    """
    Checks the correlation kernel and creates one if it is empty.

    Parameters:
        data (object): The data to be checked.
        corr_kern (object): The correlation kernel to be checked.
        param_kernel (str, optional): Parameter for the kernel type. Default is 'embedding'.
        recreate (bool, optional): Flag indicating whether to recreate the kernel. Default is False.
        know_empty (bool, optional): Flag indicating whether an empty kernel is expected. Default is False.

    Returns:
        corr_kern (object): The correlation kernel.
    """    
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
    """
    Checks if an object is an empty list.

    Parameters:
        obj (object): The object to be checked.

    Returns:
        bool: True if the object is an empty list, False otherwise.
    """    
    return isinstance(obj, list) and len(obj) == 0
    
def dictionaryRWL1SF(data, phi, corr_kern, A = [], params = {}, grannet = False, coefficients_similar_nets = [],
                     trial_num = -1, full_A = [], initial_round_free = False):
    """
    Updating A by gradient descent
    
    Parameters:
    - data: Input data
    - phi: Matrix phi
    - corr_kern: Correlation kernel
    - A: Matrix of weights (optional, default is an empty list)
    - params: Additional parameters (optional, default is an empty dictionary)
    - grannet: Flag indicating whether to use grannet (optional, default is False)
    - coefficients_similar_nets: Coefficients for similar nets (optional, default is an empty list)
    - trial_num: Trial number (optional, default is -1)
    - full_A: Full matrix A (optional, default is an empty list)
    - initial_round_free: Flag indicating whether to use initial round free (optional, default is False)
    
    Raises:
    - ValueError if the inputs are invalid
    
    Returns:
    - A: Updated matrix of weights
    - lambdas: Lambda values (shape is [n_neurons X p])
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
            raise IndexError('A should not be empty for grannet!')
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
       

                if params['likely_from'].lower() == 'gaussian': 


                    full_A[n_neuron, :, trial_num] = singleGaussNeuroInfer_grannet(data[n_neuron, :], phi, lambdas_vec, 
                                                                       full_A[n_neuron, :,:], trial_num, coefficients_similar_nets, 
                                                    params = params, l1 = params['l1'] , nonneg = params['nonneg'], 
                                                    ratio_null = 0.1, initial_round_free = initial_round_free)   
                elif params['likely_from'].lower() == 'poisson':
                    raise ValueError('future direction')
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

                    raise ValueError('future direction')
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




def is_1d(mat):
    """
    Checks if a matrix is one-dimensional.

    Parameters:
        mat (numpy.ndarray or list): The matrix to be checked.

    Returns:
        bool: True if the matrix is one-dimensional, False otherwise.

    Raises:
        ValueError: If the input matrix is not a numpy array or a list.
    """    
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



def normalizeDictionary(D, cutoff = 1):
    """
    Normalize the dictionary by dividing each column by its norm.
    
    Parameters:
        D (numpy.ndarray): The dictionary.
        cutoff (float, optional): Cutoff value for the norm. Columns with norm below this value will be set to zero. Default is 1.
    
    Returns:
        numpy.ndarray: The normalized dictionary.
    """

    D_norms = np.sqrt(np.sum(D**2,0))       # Get the norms of the dictionary elements 
    D       = D @ np.diag(1/(D_norms*(D_norms>cutoff)/cutoff+(D_norms<=cutoff))); # Re-normalize the basis
    return D

    
def dictionary_update(dict_old, A, data, step_s, GD_type = 'norm', params ={}):    
    """
    Update the dictionary using gradient descent optimization.
    
    Parameters:
        dict_old (numpy.ndarray): The old dictionary.
        A (numpy.ndarray): The matrix A.
        data (numpy.ndarray): The data.
        step_s (float): The step size.
        GD_type (str, optional): The type of gradient descent. Default is 'norm'.
        params (dict, optional): Additional parameters. Default is an empty dictionary.
    
    Returns:
        numpy.ndarray: The updated dictionary.
    """    
    if params['likely_from'].lower() == 'gaussian':
        dict_new = takeGDStep(dict_old, A, data, step_s, GD_type, params)
    else:
        raise ValueError('undefined yet. SiBBlInGS is only gaussian')
    if not params.get('normalizeSpatial'):
        dict_new = normalizeDictionary(dict_new,1)                            # Normalize the dictionary

    dict_new[np.isnan(dict_new)] = 0
    if np.mean(dict_new) < 1e-9:
        dict_new += np.random.rand(*dict_new.shape)
    return dict_new
    

def takeGDStep(dict_old, A, data, step_s, GD_type = 'norm', params ={}):
    """
    Take a gradient descent step to update the dictionary.

    Parameters:
        dict_old (numpy.ndarray): The old dictionary.
        A (numpy.ndarray): The matrix A.
        data (numpy.ndarray): The data.
        step_s (float): The step size.
        GD_type (str, optional): The type of gradient descent. Default is 'norm'.
        params (dict, optional): Additional parameters. Default is an empty dictionary.

    Raises:
        ValueError: If the GD_type is not defined.

    Returns:
        numpy.ndarray: The updated dictionary.
    """
    l2 = params['l2'] # Frob. on dict
    l3 = params['l3'] # smoothness 
    l4 = params['l4'] # correaltions between dict elements
    l5 =  params['l5'] 
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
        dict_new = update_LSwithCont(dict_old, A, data, l3, l2 = l2, params = params);
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
        dict_new = update_FullLsCor(dict_old, A, data, l2, l3, l4, l5, params)
    elif GD_type =='sparse_deconv':
        dict_new   = sparseDeconvDictEst(dict_old,data,A,params.h,params); # This is a more involved function and needs its own function
    else:
        raise ValueError('GD_Type %s is not defined in the takeGDstep function'%GD_type)        

    return dict_new
    
    
    
    
    
      
    
  
    
    
    
    
def dictInitialize(phi = [], shape_dict = [], norm_type = 'unit', to_norm = True, params = {},  to_norm_mat = False):

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
        if to_norm_mat:
            return np.abs(norm_mat(phi, type_norm = norm_type, to_norm = to_norm))
        else:
            return np.abs(phi)
    else:
       
        phi = createMat(shape_dict, params)


        return dictInitialize(phi, shape_dict, norm_type, to_norm,  params)

    
def createMat(shape_dict,  params = params_default ):
    """
    Create a matrix with a specified shape and distribution
    
    Parameters:
    - shape_dict: Dictionary containing the shape of the matrix (keys: 'rows', 'cols')
    - params: Additional parameters (optional, default is params_default)
    
    Raises:
    - ValueError if the distribution is unknown
    
    Returns:
    - mat: Created matrix
    """

    params = {**{'mu':0, 'std': 1}, **params}
    dist = params['dist_init']

    if dist == 'uniform':
        uniform_vals = params['uniform_vals']
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
   

def is_pos_def(x):
    """
    Check if a matrix is positive definite.

    Parameters:
        x (numpy.ndarray): The matrix to check.

    Returns:
        bool: True if the matrix is positive definite, False otherwise.
    """
    return np.all(np.linalg.eigvals(x) > 0)


def singleGaussNeuroInfer(lambdas_vec, data, phi, l1,  nonneg, A = [], 
                          ratio_null = 0.1, params = {}, grannet = False):
    """
    Perform weighted LASSO inference for a single vector.

    Parameters:
    ----------
    lambdas_vec : array-like
        Vector of weights.
    data : array-like
        Data vector.
    phi : array-like
        Dictionary matrix.
    l1 : float
        L1 regularization parameter.
    nonneg : bool
        Flag indicating whether non-negativity constraint should be enforced.
    A : array-like, optional
        Initial coefficient vector.
    ratio_null : float, optional
        Ratio threshold for nullifying coefficients.
    params : dict, optional
        Additional parameters.
    grannet : bool, optional
        Flag indicating whether to use the GRANNET solver.

    Returns:
    -------
    A : array-like
        Coefficient vector resulting from the inference.
    """    
    # Use quadprog  to solve the weighted LASSO problem for a single vector
    #include_Ai = params['grannet_params']['include_Ai']

    if phi.shape[1] != len(lambdas_vec):
        raise ValueError('Dimension mismatch!')  
   
    # Set up problem
    data = data.flatten()                                           # Make sure time-trace is a column vector
    lambdas_vec = lambdas_vec.flatten()                             # Make sure weight vector is a column vector
    p      = len(lambdas_vec)                                       # Get the numner of dictionary atoms
    
    if  params['solver'].lower() == 'spgl1':
         l1 *= lambdas_vec.mean()                                                       ## Run the weighted LASSO to get the coefficients    
    if len(data) == 0 or np.sum(data**2) == 0:
        A = np.zeros(p)                                             # This is the trivial solution to generate all zeros linearly.
        print('data activity is zero for this neuron and state')
    
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
                    # Solve the weighted LASSO using TFOCS and a modified linear operator
                    if params['norm_by_lambdas_vec']:
                        A = A.flatten()/lambdas_vec.flatten();              # Re-normalize to get weighted LASSO values
                        #  consider changing to oscar like they did here https://github.com/vene/pyowl/blob/master/pyowl.py 
                else:
                    phi_T_phi = phi.T @ phi
                    
                A = solve_qp(2*(phi.T @ phi),-2*phi.T @ data+l1*lambdas_vec, 
                             solver = params['solver_qp'] )         # Use quadratic programming to solve the non-negative LASSO

                if np.isnan(A).any(): 
                    raise ValueError('There are nan values is A')
          
        else:
           A = solve_Lasso_style(phi, data, l1, [], params = params, random_state = 0).flatten()
           # Solve the weighted LASSO using TFOCS and a modified linear operator
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
    Perform weighted LASSO inference for a single vector using GRANNET.

    Parameters:
    ----------
    data : array-like
        Data vector.
    phi : array-like
        Dictionary matrix.
    lambdas_vec : array-like
        Vector with p elements.
    full_A_2d : array-like
        Full coefficient matrix.
    trial_num : int
        Index of the trial.
    coefficients_similar_nets : array-like
        Coefficients from similar networks.
    params : dict, optional
        Additional parameters. Default is params_default.
    l1 : scalar, optional
        L1 regularization parameter. Default is 1.
    nonneg : bool, optional
        Flag indicating whether non-negativity constraint should be enforced. Default is False.
    ratio_null : float, optional
        Ratio threshold for nullifying coefficients. Default is 0.1.
    initial_round_free : bool, optional
        Flag indicating whether initial round is free. Default is True.

    Raises:
    ------
    ValueError
        If there is a dimension mismatch.

    Returns:
    -------
    A : array-like
        Coefficient vector resulting from the inference. Representing the BBs.
    """

    include_Ai = params['grannet_params']['include_Ai']
    labels_indicative = params['grannet_params']['labels_indicative']
    if full_A_2d[:,trial_num].sum() == 0 and initial_round_free:
        return singleGaussNeuroInfer(lambdas_vec, data, phi, l1,  nonneg, A = full_A_2d[:,trial_num], 
                                  ratio_null =  ratio_null, params = params)


    #if labels_indicative:
    if len(coefficients_similar_nets.shape) == 3:
        coefficients_similar_nets_2d = coefficients_similar_nets[trial_num] # trial num X num nets
    else:
        coefficients_similar_nets_2d = coefficients_similar_nets

        
    # check if lambas_vec has p elements
    if phi.shape[1] != len(lambdas_vec.flatten()):        raise ValueError('Dimension mismatch!')   
    p   = phi.shape[1]                                    # Get the numner of dictionary atoms     

                                                                    # Run the weighted LASSO to get the coefficients    
    if len(data) == 0 or np.sum(data**2) == 0:    
        print('neuron  has 0 activity for trial %d'%trial_num)

        
    else:

        new_data = np.vstack([data.reshape((-1,1)),
                              np.vstack([(coefficients_similar_nets_2d[trial_counter,:].reshape((-1,1)) * full_A_2d[:,trial_counter].reshape((-1,1)) * lambdas_vec.reshape((-1,1))     ).reshape((-1,1))
                                 if (trial_counter != trial_num or include_Ai) 
                                 else 0*full_A_2d[:,trial_counter].reshape((-1,1))
                                for trial_counter in range(coefficients_similar_nets_2d.shape[0])])]).reshape((-1,1))
        new_phi = np.vstack([phi/lambdas_vec.reshape((1,-1))] + [np.diag(coefficients_similar_nets_2d[trial_counter,:])
                                     if  (trial_counter != trial_num or include_Ai) else np.zeros((phi.shape[1],phi.shape[1])) 
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
           
           if params['norm_by_lambdas_vec']:
               A = A.flatten()/lambdas_vec.flatten();              # Re-normalize to get weighted LASSO values
               #  consider changing to oscar like here https://github.com/vene/pyowl/blob/master/pyowl.py 
           full_A_2d[:,trial_num] = A
    if params['nullify_some']:
        full_A_2d[:,trial_num][full_A_2d[:,trial_num] < ratio_null*np.max(full_A_2d[:,trial_num])] = 0;    
    return full_A_2d[:,trial_num]


def solve_Lasso_style(A, b, l1, x0, params = {}, lasso_params = {},random_state = 0):
    """
    Solves the l1-regularized least squares problem:
        minimize (1/2)*norm(A * x - b)^2 + l1 * norm(x, 1)

    Parameters:
    ----------
    A : array-like
        Coefficient matrix.
    b : array-like
        Target vector.
    l1 : float
        Scalar between 0 and 1, describing the regularization term on the coefficients.
    x0 : array-like
        Initial guess for the solution.
    params : dict, optional
        Additional parameters. Default is {}.
    lasso_params : dict, optional
        Additional parameters for the Lasso solver. Default is {}.
    random_state : int, optional
        Random state for reproducibility. Default is 0.

    Raises:
    ------
    NameError
        If an unknown solver is specified.

    Returns:
    -------
    x : np.ndarray
        The solution for min (1/2)*norm(A * x - b)^2 + l1 * norm(x, 1).

    lasso_options:
        - 'inv' (least squares)
        - 'lasso' (sklearn lasso)
        - 'fista' (https://pylops.readthedocs.io/en/latest/api/generated/pylops.optimization.sparsity.FISTA.html)
        - 'omp' (https://pylops.readthedocs.io/en/latest/gallery/plot_ista.html#sphx-glr-gallery-plot-ista-py)
        - 'ista' (https://pylops.readthedocs.io/en/latest/api/generated/pylops.optimization.sparsity.ISTA.html)
        - 'IRLS' (https://pylops.readthedocs.io/en/latest/api/generated/pylops.optimization.sparsity.IRLS.html)
        - 'spgl1' (https://pylops.readthedocs.io/en/latest/api/generated/pylops.optimization.sparsity.SPGL1.html)

        - Refers to the way the coefficients should be calculated (inv -> no l1 regularization).
    """
    if np.isnan(A).any():
        print(A)
        input('ok? solve_Lasso_style')
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
    
      #input('ok?')
      clf.fit(A,b.flatten() )     #reshape((-1,1))
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
        
      
        x = pylops.optimization.sparsity.IRLS(Aop, b.flatten(),  nouter=50, espI = l1)[0]      
    else:     
      raise NameError('Unknown update c type')  
    return x


def updateLambdasMat(A, corr_kern, beta, params ):
    """
    Update the weight updates (lambdas) used in a sparse coding algorithm.

    Parameters
    ----------
    A : np.ndarray
        Dictionary matrix of shape (n_neurons, p), where n_neurons is the number of neurons and p is the number of features.
    corr_kern : np.ndarray
        Correlation kernel matrix of shape (n_neurons, n_neurons).
    beta : float
        Regularization parameter.
    params : dict
        Additional parameters for the function. Possible keys are:
        - 'epsilon': Numerator of the weight updates. Can be a scalar, a list of length 1, or a matrix of shape (n_neurons, p).
        - 'updateEmbed': Boolean value indicating whether to recalculate the graph based on the current estimate of coefficients.
        - 'mask': Not used in this function.

    Returns
    -------
    lambdas : np.ndarray
        Weight updates (lambdas) of shape (n_neurons, p).

    Raises
    ------
    ValueError
        If an invalid option is provided for any of the parameters.

    Notes
    -----
    The function updates the weight updates (lambdas) used in a sparse coding algorithm based on the provided inputs. The weight updates are calculated using the following formula:
        lambdas = epsilon / (beta + A + params['zeta'] * corr_kern @ A)

    If 'updateEmbed' is True, the graph is recalculated based on the current estimate of coefficients. However, this functionality is not implemented yet.

    If the numerator of the weight updates (epsilon) is a scalar or a list of length 1, the weight updates are calculated using the entire dictionary and correlation kernel.

    If the numerator of the weight updates is a matrix of shape (n_neurons, p), the weight updates are calculated based on the corresponding elements of the dictionary and correlation kernel.

    The function raises a ValueError if an invalid option is provided for any of the parameters.

    """
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
            lambdas = params['epsilon']/(beta + A + params['zeta']*corr_kern @ A)                            #    - Calculate the wright updates tau/(beta + |s_i| + [P*S]_i) - calculate lambda!!!!
        elif not params['updateEmbed']:     
                                                #  - If the graph was not updated, use the original graph (in corr_kern)
            if corr_kern.shape[0] ==  n_neurons    :                                 #    - If the weight projection matrix has the same number of rows as pixels, update based on a matrix multiplication                     

                lambdas = params['epsilon']/(beta + A + params['zeta']*corr_kern @ A);                #      - Calculate the wright updates tau/(beta + |s_i| + [P*S]_i)
            else:
                raise ValueError('This case is not defined yet') #future


    elif len(params['epsilon'].flatten()) == p:            # If the numerator of the weight updates is the size of the dictionary (i.e., one tau per dictioary element)...
   
        if params['updateEmbed'] :                                            #  - If the graph was updated, use the new graph (i.e., P)
            lambdas = params['epsilon'].reshape((1,-1))/(beta + A + params['zeta']*corr_kern @ A)         #    - Calculate the wright updates tau/(beta + |s_i| + [P*S]_i)
        else   :                                       #  - If the graph was not updated, use the original graph (in corr_kern)
            if corr_kern.shape[0] == n_neurons :                                    #    - If the weight projection matrix has the same number of rows as pixels, update based on a matrix multiplication
                lambdas =  params['epsilon'].reshape((1,-1))/(beta + A + params['zeta']*corr_kern @ A) #    - Calculate the wright updates tau/(beta + |s_i| + [P*S]_i)
            else   :
                raise ValueError('Invalid kernel shape') #future                #  - Otherwise, the graph was updated; use the original graph (in corr_kern)
 
    elif params['epsilon'].shape[0] == A.shape[0] and params['epsilon'].shape[1] == A.shape[1]: #future
    
        raise ValueError('This option is not available yet')

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

    elif isinstance(mov, np.ndarray) and len(np.shape(mov)) == 3:
      
        to_d2_return = np.hstack([mov[:,:,frame_num].flatten().reshape((-1,1)) for frame_num in range(mov.shape[2])])  
     
        return   to_d2_return
    else:
        raise ValueError('Unrecognized dimensions for mov (cannot change its dimensions to 2d)')
    
def MovToD2_grannet(data): 
    """
    Convert a 3D data array to a 2D array.
    
    Parameters
    ----------
    data : np.ndarray or list
        3D data array of shape (N, T, k) or list of length N.
    
    Returns
    -------
    np.ndarray
        2D data array of shape (N, T*k).
    
    Raises
    ------
    ValueError
        If the data type is not supported.
    
    Notes
    -----
    The function converts a 3D data array to a 2D array by stacking the data along the third dimension. The resulting array has a shape of (N, T*k), where N is the number of samples, T is the number of time points, and k is the number of features.
    
    If the data type is not supported (not an ndarray or list), a ValueError is raised.
    
    """
    # data 3d is expected to be N X T x k

    if isinstance(data, np.ndarray):
        return np.hstack([data[:,:,k] for k in range(data.shape[2])])
    elif isinstance(data,list):
        return np.hstack([data[k] for k in range(len(data))])
    else:
        raise ValueError('unsupported type %s'%str(type(data)))
    
    
    
def D2ToMov_inv(data_2d, shape )    :
    """
    Invert the conversion from a 2D array to a 3D data array.

    Parameters
    ----------
    data_2d : np.ndarray
        2D data array of shape (pix X pix, time).
    shape : tuple
        Shape of the original frames (pix, pix).

    Returns
    -------
    np.ndarray
        3D data array of shape (pix, pix, time).

    Notes
    -----
    The function takes a 2D data array of shape (pix X pix, time) and reshapes it into a 3D data array of shape (pix, pix, time), where pix is the number of pixels in each dimension and time is the number of time points.

    """    
    # get_data in shape of (pix X pix) X time
    # give pixels X pixels X time 3d 
    return np.dstack([data_2d[:,k].reshape(shape) for k in range(data_2d.shape[1])])
    
    
def normalize_to_95_perc(d): 
    """
    Normalize data to the 99.999th percentile.

    Parameters
    ----------
    d : np.ndarray
        Data array.

    Returns
    -------
    np.ndarray
        Normalized data array.

    Notes
    -----
    The function normalizes the input data array by dividing it by the 95th percentile value along each feature dimension. The normalization is applied independently to each feature.

    """
    stacked = MovToD2_grannet(d)
    ratio_norm = np.percentile(stacked,99.999,axis = 1); print(ratio_norm)
    
    d = d/ ratio_norm.reshape((-1,1,1))
    return d
        
    
def D2ToMov(mov, frameShape, type_return = 'array'):
    """
    Convert a 2D data array to a list or np.ndarray of frames.

    Parameters
    ----------
    mov : np.ndarray
        2D data array of shape (N, time).
    frameShape : tuple
        Shape of each frame (pix, pix).
    type_return : str, optional
        Type of the return value. Can be 'array' or 'list'. Default is 'array'.

    Returns
    -------
    list or np.ndarray
        Frames with shape (frameShape, time).

    Raises
    ------
    ValueError
        If the dimensions do not fit.

    Notes
    -----
    The function converts a 2D data array to a list or np.ndarray of frames. The input mov should have a shape of (N, time), where N is the number of pixels in each frame and time is the number of time points. The output frames have a shape of (frameShape, time), where frameShape is the shape of each frame.

    The type_return parameter specifies whether the output should be a list or np.ndarray. If an invalid type is provided, a ValueError is raised.

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
    """
    Perform synthetic evaluation by comparing the ground truth dynamics with the results.

    Parameters
    ----------
    full_A : np.ndarray
        Full dictionary of shape (N, M, T) containing the estimated dictionary atoms.
    full_phi : np.ndarray
        Full sparse code matrix of shape (M, T, N) containing the estimated sparse codes.
    real_full_A : dict, optional
        Dictionary of ground truth dictionary atoms. Default is an empty dictionary ({}).
    real_full_phi : dict, optional
        Dictionary of ground truth sparse codes. Default is an empty dictionary ({}).

    Returns
    -------
    np.ndarray
        Ordered full sparse code matrix of shape (M, T, N) based on matching with ground truth.
    np.ndarray
        Ordered full dictionary of shape (N, M, T) based on matching with ground truth.

    Notes
    -----
    The function performs synthetic evaluation by comparing the ground truth dynamics (real_full_A and real_full_phi) with the estimated dynamics (full_A and full_phi). It matches the estimated dynamics with the ground truth dynamics based on time ordering.

    If the ground truth dynamics are not provided (real_full_A and real_full_phi are empty dictionaries), the function loads them from a file ('grannet_synth_results_march_2023.npy').

    The output is the ordered full sparse code matrix (full_phi_ordered) and the ordered full dictionary (full_A_ordered).

    """    
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
    Calculate the absolute value of the correlation coefficient between two vectors.

    Parameters
    ----------
    v1 : np.ndarray
        First vector.
    v2 : np.ndarray
        Second vector.
    to_abs : bool, optional
        Whether to return the absolute value of the correlation coefficient. Default is True.

    Returns
    -------
    float
        Correlation coefficient between v1 and v2.

    Notes
    -----
    The function calculates the correlation coefficient between two vectors using np.corrcoef. By default, it returns the absolute value of the correlation coefficient. If to_abs is set to False, it returns the raw correlation coefficient.

    """
    corr = np.corrcoef(v1.flatten(),v2.flatten())
    if to_abs:
        return np.abs(corr[0,1])
    return corr[0,1]
      
      
      
    
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
            phi1 = phi1.T
            turnphi = True
        elif  phi1.shape[1] == A.shape[0]: # here A is not ok
            A = A.T
            phi1 = phi1.T
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
        corr_mat[inds[0],:] = 0 
        corr_mat[:,inds[1]] = 0

    
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
    


    if not grannet or len(data.shape) == 2:
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
        


    return K
    
def mkDataGraph_grannet(data, params = {}, reduceDim = False, reduceDimParams = {}, graph_function = 'gaussian',
                K_sym  = True, use_former = False, 
                data_name = 'none', toNormRows = True,  graph_params = params_default['graph_params'],
                grannet = False):
    """
    Generate a kernel matrix for graph-based analysis using the Grannet method
    
    Parameters:
    - data: 3D array of data (neurons x time x trials)
    - params: Additional parameters (optional, default is {})
    - reduceDim: Boolean flag indicating whether to reduce dimensionality (optional, default is False)
    - reduceDimParams: Parameters for dimensionality reduction (optional, default is {})
    - graph_function: Graph function to use (optional, default is 'gaussian')
    - K_sym: Boolean flag indicating whether to symmetrize the kernel matrix (optional, default is True)
    - use_former: Boolean flag indicating whether to use the former kernel matrix (optional, default is False)
    - data_name: Name of the data (optional, default is 'none')
    - toNormRows: Boolean flag indicating whether to normalize rows of the kernel matrix (optional, default is True)
    - graph_params: Graph parameters (optional, default is params_default['graph_params'])
    - grannet: Boolean flag indicating whether to use Grannet method (optional, default is False)
    
    Returns:
    - K: Kernel matrix (3D array)
    """    
                                 
    terms_and_labels_trend = np.load('grannet_trends_for_jupyter_results_march_2023.npy', allow_pickle=True).item()
    terms =  terms_and_labels_trend['terms']
    labels = terms_and_labels_trend['labels']
    
    path_exp = r'./' # PATH TO SAVE, PLEASE CHANGE TO YOUR PREFERENCE.

    non_zeros = params['n_neighbors'] + 1
    if not grannet or (grannet and graph_params['kernel_grannet_type'] == 'ind'):  # and checkEmptyList(data):
        raise ValueError('should not use this function in this case!')
    else: # in case of grannet
        """
        1) create kernel for each !!!!!!!!!!!
        """
        if isinstance(data, np.ndarray) or ( isinstance(data, list) and np.array([data[0].shape[1] ==data_i.shape[1] for data_i in data]).all() ):
        
            kernels_inds = np.dstack([cal_dist(k, data, graph_params = params['graph_params'], 
                                    grannet = True, distance_metric = params['distance_metric'])
                            for k in range(data.shape[2])])
        else:
            kernels_inds = np.dstack([cal_dist(0,data_i, graph_params = params['graph_params'], 
                                    grannet = True, distance_metric = params['distance_metric'])
                            for data_i in data])

        """
        2)     apply average or weighting   !
        """
        if graph_params['kernel_grannet_type'] == "combination" :
            """
            in this case we need to calculate the shared graph
            """
            shared_kernel = cal_dist(0, MovToD2_grannet(data), graph_params = params['graph_params'], 
                                    grannet = True, distance_metric = params['distance_metric']) 
            print('finished stage 2: calculated shared ker')
            

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
        3) apply knn +  sym !!!!!!!!!!!
        """
        K = take_NN_given_dist_mat(dist_mat,non_zeros, K_sym = True, include_abs = True,  toNormRows = True)     
        K = np.dstack([normalize_K(K[:,:,k], toNormRows = toNormRows) for k in range(K.shape[2])])
    
        if K_sym:
            K = np.dstack([(K[:,:,i] + K[:,:,i].T)/2 for i in range(K.shape[2])])

        """
        4) normalize kernel  !!!!!!!!!!!
        """   
        K = np.dstack([normalize_K(K[:,:,k], toNormRows = toNormRows) for k in range(K.shape[2])])
        try:
            plt.figure(figsize = (20,9))
            ss = int(str(datetime2.now()).split('.')[-1])
            sns.heatmap(pd.DataFrame(K[:,:,0], terms, terms), robust = True)
            plt.savefig(path_exp + os.sep + 'try_hea%d.png'%ss)
            plt.show()
            plt.close()
        except:
            print('did not print graph')
            
        return K
    
def normalize_K(K, toNormRows = True):
    """
    Normalize the kernel matrix K
    
    Parameters:
    - K: Kernel matrix
    - toNormRows: Boolean flag indicating whether to normalize rows of the kernel matrix (optional, default is True)
    
    Returns:
    - K: Normalized kernel matrix
    """    
    K = K - np.diag(np.diag(K) )
    if toNormRows:
        K = K/K.sum(1).reshape((-1,1))         
    
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

        
        
        
        


def dist_vecs(vec1, vec2, distance_metric = 'euclidean'):
    """
    Calculate the distance between two vectors vec1 and vec2.

    Parameters:
    - vec1: First vector
    - vec2: Second vector
    - distance_metric: Distance metric to use (optional, default is 'euclidean')

    Returns:
    - Distance between vec1 and vec2
    """    
    if distance_metric == 'euclidean':
        return np.sum((vec1.flatten() - vec2.flatten())**2)
    else:
        raise ValueError('not defined yet :(')
    
    

    
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
    data_3d can be a matrix of N X T X k or a list of N x T matrices
    this function is only for the case where graph_params['kernel_grannet_type'] == "averaged":
    THIS FUNCTION IS CALLED AFTER!!! WE FOUND THE INDIVIDUAL KERNELS
    
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
    w_vec = np.array([1, w])
    w_vec = w_vec / np.sum(w_vec)
   
    
    """
    averaging
    """    
    data_3d_weighted = [np.sum(np.dstack([data_3d_list[k_weight]*w_k[k_weight] for k_weight, w_k in enumerate(normalize_w_with_i(w_vec, k) ) ]), 2)        
        for k in range(len(data_3d_list))]   
        
    if return_type == 'array':
        data_3d_weighted = np.dstack(data_3d_weighted)  
        
    return data_3d_weighted 


def take_NN_given_dist_mat(dist_mat,non_zeros, K_sym = True, include_abs = True,  toNormRows = True):
    """
    Calculate the k-nearest neighbors (KNN) given a distance matrix.

    Parameters:
    - dist_mat: Distance matrix (N x N)
    - non_zeros: Number of nearest neighbors to consider
    - K_sym: Flag to indicate whether to symmetrize the KNN matrix (optional, default is True)
    - include_abs: Flag to indicate whether to include the absolute values in the KNN matrix (optional, default is True)
    - toNormRows: Flag to indicate whether to normalize the rows of the KNN matrix (optional, default is True)

    Returns:
    - KNN matrix (N x N)
    """    
    # given N X N matrices of the euclidean distance between neurons, calculate the 
    if isinstance(dist_mat, list) or len(dist_mat.shape) == 3:
        if isinstance(dist_mat, list):
            dist_mat = np.dstack(dist_mat)
        return np.dstack([take_NN_given_dist_mat(dist_mat[:,:,k],non_zeros) for k in range(dist_mat.shape[2])])
    
    corr_kern  = gaussian_vals(dist_mat, std = np.median(dist_mat[dist_mat != 0 ]))
    K = np.vstack([hard_thres_on_vec(corr_kern[n], non_zeros, include_abs).reshape((1,-1)) 
               for n in range(corr_kern.shape[0])])

        
    return K
    

def hard_thres_on_A(A_2d, non_zeros, direction = 1):
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
    if direction == 0:
        A_ret = np.hstack([hard_thres_on_vec(A_2d[:,t], non_zeros).reshape((-1,1)) for t in range(A_2d.shape[1])])
    if direction == 1:
        A_ret = np.vstack([hard_thres_on_vec(A_2d[t,:], non_zeros).reshape((1,-1)) for t in range(A_2d.shape[0])])
    if A_ret.shape != A_2d.shape:
        raise ValueError('A shapes must be identical but %s and %s'%(str(A_2d.shape), str(A_ret.shape)))

    
    return A_ret
        
        
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
    this function takes a combination if individual and shared kernel
    w is the weight of the joint graph
    data_3d can be a matrix of N X T X k or a list of N x T matrices
    
    shared kernel = kerenel of all states
    w decides the weight unless 0    
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
    Find nearest neighbors for a given dataset.

    Parameters:
    - data: Dataset (N x D)
    - params: Parameters dictionary
    - data_name: Name of the dataset (optional, default is 'none')
    - use_former: Flag to indicate whether to use previously computed nearest neighbors (optional, default is False)
    - addi: Additional string to append to the saved file name (optional, default is '_knn')
    - to_save: Flag to indicate whether to save the nearest neighbors dictionary (optional, default is True)

    Returns:
    - Nearest neighbors dictionary

    this one is ONLY for graft (not grannet/sibblings) or the grannet case of 'ind' (i.e. when params[graph_params]['kernel_grannet_type'] is ind)
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

    if not is_mask: mat += np.inf # (check_effect)

    rows = np.arange(inds.shape[0])   
    for row in rows:
        col_column = inds[row,:]

        for col_count, col in enumerate(col_column):

            if isinstance(defVal, np.ndarray):

                mat[row,col] = defVal[row,col_count]
            else:
                mat[row,col] = defVal    

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


        g = np.exp(-((mat-mean)/std)**power)
        if norm: return g/np.sum(g)

    elif dimensions == 2:

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
        
    

def update_GDiters(dict_old, A, data, step_s, params):
    """
    Take a step in the negative gradient of the basis: Minimizing the energy E = ||x-Da||_2^2 + lambda*||a||_1^2

    Parameters:
    - dict_old: Dictionary matrix (K x M)
    - A: Activation matrix (N x K)
    - data: Data matrix (N x M)
    - step_s: Step size
    - params: Parameters dictionary

    Returns:
    - Updated dictionary matrix (K x M)
    """
    for index2 in range(params.get('GD_iters')):
        # Update The basis matrix

        dict_old = dict_old + (step_s/A.shape[0])*((data.T - dict_old @ A.T) @ A) 

    # This part is basically the same, only for the hyyperspectral, care needs to be taken to saturate at 0,
    # so that no negative relflectances are learned. 
    if params.get('nonneg'):
        dict_old[dict_old < 0] = 0 + epsilon
        if np.sum(dict_old) ==0:
            raise ValueError('sum should not be 0')
    return dict_old









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
        dict_new = np.zeros(dict_old.shape)                                  # Initialize the dictionary
        n_times = dict_old.shape[0]
        for  t in range(n_times):
            dict_new[t,:] = nnls(A, data[:,t]) # Solve the least-squares via a nonnegative program on a per-dictionary level                   
    else:
        dict_new = data.T @ np.pinv(A);                                         # Solve the least-squares via an inverse

    return  dict_new


def  update_LSwithCont(dict_old, A, data, l3, l2, params):    
    """
    Minimizing the energy: E = ||X - DA||_2^2 + l3 * ||D_old - D||_F^2 via D = [X;D_old] * pinv([A;I])

    Parameters:
    - dict_old: Dictionary matrix (K x M)
    - A: Activation matrix (N x K)
    - data: Data matrix (N x M)
    - l3: Regularization parameter for the difference between D_old and D
    - l2: Regularization parameter for the Frobenius norm of D
    - params: Parameters dictionary

    Returns:
    - Updated dictionary matrix (K x M)
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
    Minimizing the energy: E = ||data.T - DA.T||_2^2 + l3 * ||D_old - D||_F^2 + l2 * ||D||_F^2, via phi = [data.T;phi_old] * pinv([A.T;I])

    Parameters:
    - dict_old: Dictionary matrix (K x M)
    - A: Activation matrix (N x K)
    - data: Data matrix (N x M)
    - l2: Regularization parameter for the Frobenius norm of D
    - l3: Regularization parameter for the difference between D_old and D
    - params: Parameters dictionary

    Returns:
    - Updated dictionary matrix (K x M)
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
        dict_new = np.vstack([data,l3*dict_old.T,l2*dict_old]) @  np.linalg.pinv(np.vstack([A.T,l3*np.eye(n_neurons),np.zeros((n_neurons, n_neurons))])); # Solve the least-squares via an inverse
    return  dict_new
     
def  update_LSwithForb(dict_old, A, data, l2, params):
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
    
def  update_FullLsCor(dict_old, A, data, l2, l3, l4, l5, params,  epsilon_D = 0.001):
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
        
    link of regularization to paper:
        code - paper
        l1 - lambda
        l2 - gamma 1 (frob norm)
        l3 - gamma 3 continuation
        l4 - gamma 2 (diag - correlations)
        l5 - gamma 4 time continues

    """
    
    dict_new = np.zeros(dict_old.shape);                                  # Initialize the dictionary
 
    n_times    = dict_old.shape[0]   
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
        p = dict_old.shape[1]
        l2_cols = np.expand_dims(np.sqrt((dict_old**2).sum(0)),0)

        D =  np.outer(l2_cols, l2_cols)
        D[D == 0] = epsilon_D
        M_tilda = np.vstack([np.zeros((A.shape[1], A.shape[1])) , l4/D, (l2  + l3 + l5)*np.eye(p) - l4*np.diag(np.diag(1/D)) ])
       
        for t in range(n_times): # future step - make it more efficients by contcatinating     

            if t == 0:
                # no l5                   
                M_tilda_0 = np.vstack([np.zeros((A.shape[1], A.shape[1])) , l4/D, (l2 + l3 )*np.eye(p)  - l4*np.diag(np.diag(1/D)) ])
                try:
                  
                    Y_tilda_0 = np.vstack([(np.linalg.pinv(A) @ data[:,t].reshape((-1,1))).reshape((-1,1))   , np.zeros((p,1)).reshape((-1,1)) , l3*dict_old[t,:].reshape((-1,1))   ])
                except:
                    print(A)
                    input('A ok?')
                work = False
                while not work:
                    try:
                        np.linalg.pinv(M_tilda_0)
                        work = True
                    except:
               
                        M_tilda_0 += np.random.rand(*M_tilda_0.shape)*0.05
                          
               
                if not  params['phi_positive']:
                    dict_new[0,:] = (np.linalg.pinv(M_tilda_0) @  Y_tilda_0.reshape((-1,1)) ).flatten()
                else:
                    dict_new[0,:] = nnls(M_tilda_0, Y_tilda_0.flatten())[0]
            else:
                Y_tilda = np.vstack([( np.linalg.pinv(A) @ data[:,t].reshape((-1,1))).reshape((-1,1))   , np.zeros((p,1)).reshape((-1,1)) , l3*dict_old[t,:].reshape((-1,1))  + l5*dict_new[t-1,:].reshape((-1,1))  ])
                if not  params['phi_positive']:
                    dict_new[t,:] = (np.linalg.pinv(M_tilda) @  Y_tilda.reshape((-1,1)) ).flatten()
                else:
                    dict_new[t,:] = nnls(M_tilda, Y_tilda.flatten())[0]


    return dict_new

def sparseDeconvDictEst(dict_old, A, data, l2, params):
    """

    
    
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
    

    raise ValueError('Function is currently not available. Please change the "GD_type" from "sparse_deconv"')
    pass




def boots_check_corr(phi, s_boots = 10, type_boots = 'within', labels = []):
    """
            
    phi is time X p X trials
    
    return   repeat X NET X state
    
    Compute bootstrapped spectral correlation between pairs of signals over trials and/or states.

    Parameters:
    -----------
    phi : numpy.ndarray
        A 3D numpy array of shape (time, p, trials) representing the data for p signals and trials.
    s_boots : int, optional
        Number of bootstrap samples. Default is 10.
    type_boots : str, optional
        Type of bootstrap. 'within' for within-state bootstrap, 'between' for between-state bootstrap. Default is 'within'.
    labels : list or numpy.ndarray, optional
        List of length `phi.shape[2]` containing the state labels for each trial. Required for within-state bootstrap.
        Default is an empty list.

    Returns:
    --------
    corr_mat : numpy.ndarray
        A 3D numpy array of bootstrapped spectral correlations. If `type_boots` is 'within', the shape is
        (s_boots, p, n_states), where n_states is the number of unique states in `labels`. If `type_boots` is
        'between', the shape is (s_boots, p, p*(p-1)/2).
    pairs : list
        List of length `p*(p-1)/2` containing all unique pairs of signals in `phi`. Only returned if `type_boots`
        is 'between'.

    Raises:
    -------
    ValueError
        If `type_boots` is 'within' and `labels` is an empty list.
    """

    
    
    p = phi.shape[1]
    if type_boots == 'within':
        rat_mul = 1
    else:
        rat_mul = int(p*(p-1)/2)
    pairs_take = np.random.randint(len(np.unique(labels)),size = s_boots*2*rat_mul+30).reshape(-1,2)
    pairs_take = np.vstack([pairs_take[i,:] for i in range(pairs_take.shape[0]) if pairs_take[i,0] != pairs_take[i,1]])
    pairs_take = pairs_take[:s_boots*2*rat_mul,:]
    
    if type_boots == 'within':
        pairs= np.unique(labels)
        corr_mat = np.zeros((s_boots, phi.shape[1], len(np.unique(labels))))
        print(corr_mat.shape)
        if checkEmptyList(labels):
            raise ValueError('you must provide labels for within') 

        for lab_count, lab in enumerate(np.unique(labels)):
            cur_phi = phi[:,:,labels == lab]
            print(cur_phi.shape)
            for repeat in range(s_boots):
                pairs_take[repeat,0]
                phi1 = cur_phi[:,:,pairs_take[repeat,0]]
                phi2 = cur_phi[:,:,pairs_take[repeat,1]]
                for net in range(p):
                    corr_mat[repeat,net,lab_count ]  = spec_corr(phi1[:,net], phi2[:,net])
                

           
    else:
        corr_mat = np.zeros((phi.shape[0], phi.shape[1],int(p*(p-1)/2)))
        # return   repeat X NET X state
        pairs = list(itertools.combinations(range(p), 2))
        
        counter = 0
    
        for lab_count,pair in enumerate(pairs): # pairs of states
            lab = np.unique(labels)[lab_count]
            lab1 = np.unique(labels)[pair[0]]
          
            
            
            phi1full = phi[:,:,labels == lab1] # choose a trial from each 
            
            lab2 = np.unique(labels)[pair[1]]
   
            phi2full  = phi[:,:,labels == lab2]
            
            for repeat in range(s_boots):
             
                trial1 = pairs_take[counter,0]
                trial2 = pairs_take[counter,1]
                phi1 = phi1full[:,:,repeat]
                phi2 = phi2full[:,:,repeat]
                for net in range(p):
                
                    corr_mat[repeat,net,lab_count ]  = spec_corr(phi1[:,net], phi2[:,net])
                    
                
                
                counter += 1
            
            
    return corr_mat, pairs
                              



    
# FOR WHITIN STATE:
#  repeat X NET X state
    
    
    
    
    
    
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



    
    
def from_folder_to_array(path_images =  r'./' 
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
    
    
def load_image_to_array(path_images =  r'./',
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

    


#%% Working with files
    
def load_mat_file(mat_name , mat_path = '',sep = sep, squeeze_me = True,simplify_cells = True):
    """
    Function to load mat files. Useful for uploading the c. elegans data. 

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
       
        
    n = np.load(path_result, allow_pickle=True).item()
    na = n['A']/n['A'].sum(0)
    fig, axs = plt.subplots(1, na.shape[1], figsize = (20,5))
    for col_num in np.arange(na.shape[1]):
        col = na[:, col_num]
        graph_one_col(col, mapping, axs[col_num])


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




def remove_background(ax, grid = False, axis_off = True):
    """
    Modify the appearance of a 3D plot by making the background transparent, hiding the grid lines, and optionally turning off the axes.

    Parameters:
    - ax: Axes object of the 3D plot
    - grid: Boolean value indicating whether to show grid lines (default: False)
    - axis_off: Boolean value indicating whether to turn off the axes (default: True)
    """    
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    if not grid:
        ax.grid(grid)
    if axis_off:
        ax.set_axis_off()
    
        
def create_ax(ax, nums = (1,1), size = (10,10), proj = 'd2',return_fig = False,sharey = False, sharex = False, fig = []):
    #ax_copy = ax.copy()
    """
    Creates a new set of axes for a plot or returns the existing axes if provided as an argument.

    Parameters:
        ax (object or list): Existing axes object or an empty list if a new axes object needs to be created.
        nums (tuple): A tuple specifying the number of rows and columns in the subplot grid (default: (1, 1)).
        size (tuple): A tuple specifying the size of the figure in inches (default: (10, 10)).
        proj (str): Projection type, either 'd2' for 2D or 'd3' for 3D (default: 'd2').
        return_fig (bool): Boolean value indicating whether to return the figure along with the axes (default: False).
        sharey (bool): Boolean value indicating whether to share the y-axis among subplots (default: False).
        sharex (bool): Boolean value indicating whether to share the x-axis among subplots (default: False).
        fig (object or list): Existing figure object or an empty list if a new figure needs to be created.

    Returns:
        ax (object): If return_fig is False (default), returns the axes object.
        fig, ax (tuple): If return_fig is True, returns the figure and axes objects.

    """    
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
    """
    Plots a 3D line plot using the provided data on the given axes object.

    Parameters:
        mat (array-like): A 3 x N array-like object containing the x, y, and z coordinates of the points to be plotted.
        ax (object): Axes object on which the plot will be drawn.
        lw (int): Line width (default: 2).
        color (str): Line color (default: 'black').
        ls (str): Line style (default: '-').
        alpha (float): Line transparency (default: 1).

    Returns:
        None
    """    
    ax.plot(mat[0],mat[1],mat[2], color = color, lw = lw, ls = ls, alpha = alpha)



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
    



#%%  MATRIX FACTORIZATION METHODS FOR COMPARISON
"""""""""""
HERE - METHODS FOR OTHER MATRIX FACTORIZATION METHODS FOR COMPARISON
"""""""""""

    

from tensorly.decomposition import tucker, parafac, non_negative_tucker
from tensorly import random as random_tl

def run_existing_methods(data, p, methods_to_compare = ['adad_svd','hosvd','parafac','tucker', 'HOOI'],
                         params_parafac = {}, params_tucker = {}):
    """
    Runs existing tensor decomposition methods on the provided data and returns the results.

    Parameters:
        data (ndarray): The tensor data to be decomposed.
        p (int): The number of components to be extracted.
        methods_to_compare (list): List of method names to be compared (default: ['adad_svd', 'hosvd', 'parafac', 'tucker', 'HOOI']).
        params_parafac (dict): Additional parameters for the Parafac decomposition (default: {}).
        params_tucker (dict): Additional parameters for the Tucker decomposition (default: {}).

    Returns:
        results (dict): Dictionary containing the decomposition results for each method.

    """    
    # the mathods are taken from http://tensorly.org/stable/modules/api.html#module-tensorly.decomposition
    # user guide http://tensorly.org/stable/user_guide/quickstart.html#tensor-decomposition
    results = {}
    #A_adad, phi_adad = run_adad_svd(data, p)
    #results['adad'] = {'A':A_adad, 'phi':phi_adad}
    A_tucker, phi_tucker, _,_ = run_tucker(data, p = p, params_tucker = params_tucker)
    results['tucker'] = {'A':A_tucker, 'phi':phi_tucker}
    A_parafac, phi_parafac, _ = run_parafac(data, p = p, params_parafac = params_parafac)
    results['parafac'] = {'A':A_parafac, 'phi':phi_parafac}
    return results
    
    
def run_adad_svd(data, p = 10, max_TK = 1000):
    """
    Runs the ADAD-SVD tensor decomposition method on the provided data.

    Parameters:
        data (ndarray): The tensor data to be decomposed.
        p (int): The number of components to be extracted (default: 10).
        max_TK (int): Maximum number of iterations for the ADAD-SVD algorithm (default: 1000).

    Returns:
        A (ndarray): Matrix A obtained from the decomposition.
        phi (ndarray): Matrix phi obtained from the decomposition.

    """
    A, s, VT = np.linalg.svd(data[:,:,0])
    A = A[:p]
    phi = VT
    return A, phi
    
    
    
    
def run_tucker(data, p = 10, params_tucker = {}):
    """
    explanation: factors[0] is A, 
    ignoring core?! :( 
    explanation: http://tensorly.org/stable/modules/generated/tucker-function.html#tensorly.decomposition.tucker

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    p : TYPE, optional
        DESCRIPTION. The default is 10.
    params_tucker : TYPE, optional
        DESCRIPTION. The default is {}.

    Returns
    -------
     : TYPE
        DESCRIPTION.
    core : TYPE
        DESCRIPTION.
    factors : TYPE
        DESCRIPTION.

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

    factors = parafac(data, rank=p, **params_parafac)
    factors_f = factors.factors
    factors_w = factors.weights
    
    full_A = []
    full_phi = []
    
    A_parafac = factors_f[0]
    phi_base_parafac = factors_f[1]
    #print('phi_base_parafac')
    #print(phi_base_parafac.shape)
    k_parafac = factors_f[2]
    for k_spec in range(k):         
        phi_parfac = np.dstack([phi_base_parafac*k_parafac[k_spec,:].reshape((1,-1)) for k_spec in range(k)] )
    # pay attention - this is before the hard thresholding and the re-order. The components are then reordered according to temporal traces correlations
    return A_parafac, phi_parfac, factors
    
    
    
    
def save_dict_to_txt(my_dict, path_save = 'my_dict.txt'):
    """
    Saves a dictionary to a text file.

    Parameters:
        my_dict (dict): The dictionary to be saved.
        path_save (str): The path to save the text file (default: 'my_dict.txt').

    """    
    with open(path_save, "w") as f:
        # Write the dictionary to the file
        f.write(str(my_dict))
    
    # Close the file
    f.close()
    
def create_dict_of_clusters_multi_d_A(full_A, labels = [], terms = [], perc_null = 90, thres = 0):
    """
    Create a dictionary of clusters for multi-dimensional A.

    Parameters:
        full_A (ndarray): The multi-dimensional A array.
        labels (list): List of labels for the clusters (default: []).
        terms (list): List of terms (default: []).
        perc_null (int): Percentile threshold for null values (default: 90).
        thres (float or list): Threshold value(s) for cluster creation (default: 0).

    Returns:
        dict: Dictionary of clusters.
        list: List of labels.
        list: List of terms.

    """    
    if checkEmptyList(terms):
        terms = np.arange(full_A.shape[0])
    if checkEmptyList(labels):
        labels = np.arange(labels.shape[2])
    if isinstance(thres ,(list, tuple, np.ndarray)) and isinstance(thres[0] ,(list, tuple, np.ndarray)):
        return {label : create_dict_of_clusters_single_A(full_A[:,:,label_count],terms = terms, perc_null = perc_null, thres = thres[label_count]) 
                for label_count, label in enumerate(labels)}, labels, terms
    return {label : create_dict_of_clusters_single_A(full_A[:,:,label_count],terms = terms, perc_null = perc_null, thres = thres) 
            for label_count, label in enumerate(labels)}, labels, terms

def create_dict_of_clusters_single_A(A_2d,terms = [], perc_null = 80, thres = 0):
    """
    Creates a dictionary of clusters from a single 2D array A.

    Parameters:
        A_2d (ndarray): The 2D array A.
        terms (list or ndarray): List of terms corresponding to the rows of A (default: []).
        perc_null (int): The percentile threshold for considering null values (default: 80).
        thres (float or list or tuple or ndarray): Threshold value(s) for considering non-null values (default: 0).

    Returns:
        dict: Dictionary of clusters.

    Raises:
        ValueError: If 0 is present in the thres list.

    """    
    if checkEmptyList(terms):
        terms = np.arange(A_2d.shape[0])
        
    if isinstance(thres,(list, tuple, np.ndarray)) and 0 in thres:
        raise ValueError('if providing thres list, 0 should not be there. but %s'%thres)        
        
    if not isinstance(thres,(list, tuple, np.ndarray)) and   thres == 0:        
        return {'group_%d'%i:terms[np.abs(A_2d[:,i])  > np.percentile(np.abs(A_2d), perc_null)]  
                for i in np.arange(A_2d.shape[1])}
    else:
        if not isinstance(thres,(list, tuple, np.ndarray)):
            thres = [thres]*A_2d.shape[1]
        return {'group_%d'%i:terms[np.abs(A_2d[:,i])  > thres[i]]  
                for i in np.arange(A_2d.shape[1])}

    

    
    




