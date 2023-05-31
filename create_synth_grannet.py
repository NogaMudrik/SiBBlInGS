# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 09:00:08 2023

@author: ----
"""
#https://gpflow.github.io/GPflow/2.7.0/notebooks/getting_started/mean_functions.html

import numpy as np
import scipy
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import seaborn as sns
import os  
from main_functions_graft_lab2 import *
"""
EXPLANATIONS:
    
    A : N X p
    PHI:p X T
    Y : N X T    
    
"""


N = 100          # num neurons  
p = 10           # num clusters 
opts = {'n_nonzeros': [np.random.randint(1,int(N/5)) for _ in range(p)], 
        'conds':['sparse','corr_temp','orth_nets', 'corr_labels', 'similar_cont'],
        'thres_cont': 3, # threshold on similar contribution
        'corr_thres_min': 0.1, 
        'corr_thres_max': 0.6, 'max_cont_thres':10
        }



def run_create_sig_dim(n = 3, num_els = 15, n_p = 300,  seed_factor = 1):
    """
    Generates an array of signals with specified dimensions.
    
    Parameters:
    n (int, optional): The number of signals to generate. Defaults to 3.
    num_els (int, optional): The number of elements in each signal (i.e. how many sin / cos are there?). Defaults to 5.
    n_p (int, optional): The number of points in each signal. Defaults to 100.
    min_t (int, optional): The minimum time value for each signal. Defaults to 0.
    max_t (int, optional): The maximum time value for each signal. Defaults to 10.
    
    Returns:
    sig_dim: An array of signals, each represented as a 1D numpy array.
    """    
    x = np.linspace(0,10,n_p)
    sig_dim = []
    for n_sig in range(n):
        np.random.seed(n_sig*seed_factor)
        sig = np.zeros(x.shape)
        for n_els in range(num_els):
            np.random.seed(n_els + n_sig*n_els + 5*seed_factor)
            sign = np.random.choice([-1,1])
            ff = np.random.choice([np.sin, np.cos])
  
            freq = np.random.rand()*5
            bb = np.random.rand(*x.shape)
            sig += sign*ff(freq*x )
        sig_dim.append(sig)
    return np.vstack(sig_dim)


def take_HT_col(col, k):
    """
    Perform hard thresholding on a column vector.

    Args:
        col (numpy.ndarray): The input column vector.
        k (int): The number of elements to keep.

    Returns:
        numpy.ndarray: The column vector after hard thresholding.
    """
    argsort_col = np.argsort(col)[::-1][:k]
    after_HT = np.zeros_like(col)
    after_HT[argsort_col] = col[argsort_col]
    return after_HT

def init_data(N, p, opts, labels = [1,2,3],num_els = 15,  n_p = 300,  seed_factor = 1):
    """
    Initialize data for dictionary learning.

    Args:
        N (int): Number of samples.
        p (int): Number of features.
        opts (dict): Options for initialization.
        labels (list): List of labels. Defaults to [1, 2, 3].
        num_els (int): Number of elements in each signal dimension. Defaults to 15.
        n_p (int): Number of time points in each signal dimension. Defaults to 300.
        seed_factor (int): Seed factor for randomization. Defaults to 1.

    Returns:
        numpy.ndarray: The initialized matrix A.
        dict: The initialized signal dimension matrix phi.
    """    
    # create A (N x p)
    A = np.random.randn(N,p)
    A = np.hstack([take_HT_col(A[:,i], opts['n_nonzeros'][i]).reshape((-1,1)) for i in range(p)     ])
    
    if len(labels) == 0:
        # create phi.T (p X T)
        phi = run_create_sig_dim( p, num_els = 15, n_p = 300,  seed_factor = 1)        
    else:
        phi = {label: run_create_sig_dim(p, num_els, n_p ,  seed_factor*counter )     
                 for counter, label in enumerate(labels)}     
    A = no_empties(A)
    return A, phi

def update_corr_temporal(phi, thres_corr_min = 0.1, thres_corr_max = 0.9):
    """
    Update the temporal activity correlations in the signal dimension matrix phi.

    Parameters
    ----------
    phi : numpy.ndarray or dict
        The signal dimension matrix or dictionary of signal dimension matrices.
    thres_corr_min : float, optional
        The minimum threshold for temporal activity correlation. Columns with correlations below this threshold will be updated. Defaults to 0.1.
    thres_corr_max : float, optional
        The maximum threshold for temporal activity correlation. Columns with correlations above this threshold will be updated. Defaults to 0.9.

    Returns
    -------
    numpy.ndarray or dict
        The updated signal dimension matrix or dictionary of updated signal dimension matrices, with temporal activity correlations adjusted based on the specified thresholds.
    """    
    # check for low temporal activity correlations
    if isinstance(phi, dict):

        return {label:update_due_to_corr(phi_spec.T, type_return = 'minmax', type_update = 'randn', threshold = {'min':thres_corr_min,
                                                                                          'max': thres_corr_max}).T
               for label, phi_spec in phi.items()}

    else:
        return update_due_to_corr(phi.T, type_return = 'minmax',  type_update = 'randn',threshold = {'min':thres_corr_min,
                                                                                          'max': thres_corr_max}).T
    
def check_sparse(A, n_nonzeros):
    """
    Check and enforce sparsity conditions on a matrix A.

    Parameters
    ----------
    A : numpy.ndarray
        The input matrix.
    n_nonzeros : int or list or tuple
        The desired number of non-zero elements in each column of A. If an int is provided, the same number of non-zero elements will be enforced for all columns. If a list or tuple is provided, each element specifies the desired number of non-zero elements for the corresponding column.

    Returns
    -------
    numpy.ndarray
        The updated matrix A with the sparsity conditions enforced.
    """    
    # return if condition is met, n_nonzeros can be an integer of a list / array of ints
    if not isinstance(n_nonzeros, (list, tuple)):
        n_nonzeros = [n_nonzeros]*A.shape[1]
    # for i in range(A.shape[1])]):
    for i in range(A.shape[1]):
        if np.sum(A[:,i] != 0) > n_nonzeros[i]:
            b = np.where(A[:,i].flatten() == 0)[0]
            A[np.random.choice(b, b - n_nonzeros[i], reprlace = False), i] = 0
    return A 
    

def check_corr_label(phi, labels,labels_cont = False, num_back_min = 2, num_label_min = 2, wind = 10, thres_corr_back = 0.2, 
                     thres_corr_label = 0.6, return_form = 'dict', include_corr = False):
    """
    Checks the correlation between temporal traces and labels, and performs updates to ensure desired correlations.
    
    Parameters:
    phi (dict): Dictionary of p X T temporal traces matrices for different labels.
    labels (list): List of labels to be considered.
    labels_cont (bool, optional): Indicates whether labels are continuous. Defaults to False.
    num_back_min (int, optional): Minimum number of background traces. Defaults to 2.
    num_label_min (int, optional): Minimum number of label traces. Defaults to 2.
    wind (int, optional): Window size for averaging correlations. Defaults to 10.
    thres_corr_back (float, optional): Threshold for background trace correlation. Defaults to 0.2.
    thres_corr_label (float, optional): Threshold for label trace correlation. Defaults to 0.6.
    return_form (str, optional): Desired form of returned output. Defaults to 'dict'.
    include_corr: Boolean. Wether to include min corr for some traces
    Returns:
    dict or ndarray: Updated temporal traces matrix in the specified return form.
    
    Raises:
    ValueError: If the minimum number of background or label traces exceeds the total number of clusters (p).
                If the window size is too wide for correlation checks.
    """

    p = phi[labels[0]].shape[0]
    T = phi[labels[0]].shape[1] 
    if num_label_min + num_back_min > p:
        raise ValueError('min background or label > num clusters')
    if wind > T/2:
        raise ValueError('wind is too wide for check_corr_label')
        
    # here phi is a dictionary of p X T dicts
    if labels_cont: # check mutual info with each phi and label
        raise ValueError('not implemented yet for cont. labels')
    else:           # cehck correlation for each time point - mov avg. check that at least T_s exist
        phi_dstack = np.dstack([np.expand_dim(phi[label],2) for label in labels]) # this is p X T X num_labels
        
        for back in range(num_back_min): # check for low correlation with label
            wind_corr_avg = [np.inf]
            while np.max(wind_corr_avg  ) > thres_corr_back:
                cur_slice_corr = [spec_corr(phi_dstack[back,t,:].flatten(), labels)  for t in T]
                wind_corr_avg = [np.mean(cur_slice_corr[np.max([1,i - wind]) : np.min([T, i + wind])])
                           for i in range(T)  ]
                if np.max(wind_corr_avg  ) > thres_corr_back:
                    phi_dstack[back,:,:] += 0.1*run_create_sig_dim(n = len(labels) , num_els = 5, n_p = T,  seed_factor = 1).T
        if include_corr:
            
            for sig in range(num_label_min): # check for low correlation with label
                wind_corr_avg = [0]
                while np.min(wind_corr_avg  ) < thres_corr_label:
                    cur_slice_corr = [spec_corr(phi_dstack[-sig,t,:].flatten(), labels)   for t in T]
                    wind_corr_avg = [np.mean(cur_slice_corr[np.max([0,i - wind]) : np.min([T, i + wind])])
                               for i in range(T)  ]
                        if np.min(wind_corr_avg  ) < thres_corr_label:
                            phi_dstack[-sig,:,:] += run_create_sig_dim(n = len(labels) , num_els = 5, n_p = T,  seed_factor = 1).T
    if return_form == 'dict':
        return {label:phi_dstack[:,:,label_count] for label_count, label in enumerate(labels)}
    return phi_dstack
                    
def check_orthogonality(A, type_return = 'minmax', threshold = 0.5, return_arg = False):
    """
    Check the orthogonality of a matrix A based on the correlation between its components.

    Parameters
    ----------
    A : numpy.ndarray
        The input matrix.
    type_return : str, optional
        The type of orthogonality check to perform. Possible values are 'max', 'min', and 'minmax'. Defaults to 'minmax'.
    threshold : float or dict, optional
        The threshold value(s) for the orthogonality check. If a float is provided, it is used as the threshold for both 'max' and 'min' checks. If a dict is provided, it should contain 'min' and 'max' keys specifying the respective thresholds. Defaults to 0.5.
    return_arg : bool, optional
        Whether to return additional information about the orthogonality check. If True, returns the result of the check and additional arguments. Defaults to False.

    Returns
    -------
    bool or tuple
        The result of the orthogonality check. If return_arg is True, returns a tuple with the check result and additional arguments. Otherwise, returns a bool indicating the check result.
    """    
    corr_A  = np.abs(np.corrcoef(A.T) - np.eye(A.shape[1])) # correlation between components

    if type_return == 'max':
        if return_arg: return np.max(corr_A) < threshold, np.argmax(corr_A)
        return np.max(corr_A) > threshold
    elif type_return == 'min':
        if return_arg: return np.percentile(corr_A,80) > threshold, np.argmin(corr_A)
        return np.min(corr_A) < threshold
    elif type_return == 'minmax':

        return1 =   np.percentile(corr_A,80) > threshold['min'] 
        return2 =   np.max(corr_A) < threshold['max']
        if return_arg: 
            if not return1 and not return2:
                return3 = [np.argmin(corr_A), np.argmax(corr_A)]
            elif not return1:
                return3 = np.argmin(corr_A)
            elif not return2:
                return3 = np.argmax(corr_A)
            else:
                return3 = 'none, strrange'
            return return1 and return2, return3
        return return1 and return2
    
    else:
        raise ValueError('unknown type_return')

def update_due_to_corr(A, type_return = 'min', threshold = 0.5, type_update = 'reorder'): # update A if does not meet conditions
    """
    Update matrix A if it does not meet the specified orthogonality conditions.

    Parameters
    ----------
    A : numpy.ndarray
        The input matrix.
    type_return : str, optional
        The type of orthogonality check to perform. Possible values are 'min', 'max', and 'minmax'. Defaults to 'min'.
    threshold : float, optional
        The threshold value for the orthogonality check. Defaults to 0.5.
    type_update : str, optional
        The type of update to perform on A if it does not meet the orthogonality conditions. Possible values are 'reorder' and 'random'. Defaults to 'reorder'.

    Returns
    -------
    numpy.ndarray
        The updated matrix A.
    """

    if type_return not in ['min','max','minmax']:
        raise ValueError('typereturn is undefined!')
    # check orth. between A-th cols. A is N X p.
    orth_met =  check_orthogonality(A, type_return,threshold,  return_arg= False)
    p = A.shape[1]
    while not orth_met:

            

        orth_met, arg_not_met = check_orthogonality(A, type_return, threshold, return_arg= True)
       
        if not orth_met:
            if not isinstance(arg_not_met, list):
                ind_bad = np.unravel_index(arg_not_met, (p,p))     
            else:
                ind_bad = np.array(lists2list([np.unravel_index(arg_not_met_spec, (p,p)) for arg_not_met_spec in arg_not_met  ]))
            if type_update == 'reorder':
                A[:, ind_bad] = reorder_A(A[:, ind_bad])
            else:
                A[:, ind_bad] += np.random.randn(*A[:, ind_bad].shape)*0.02
            
    return A
    
def lists2list(xss)    :
    """
    Convert a list of lists into a flat list.

    Parameters
    ----------
    xss : list
        The list of lists.

    Returns
    -------
    list
        The flattened list.
    """    
    return [x for xs in xss for x in xs]  

    
def check_similar_contirubtion(phi, A):
    """
    Check the similarity of contributions between temporal profiles phi and neural maps A.
    
    Parameters
    ----------
    phi : numpy.ndarray
        The temporal profiles. Shape: p x T.
    A : numpy.ndarray
        The neural maps. Shape: N x p.
    
    Returns
    -------
    float, tuple
        The maximum contribution difference and the indices of the maximum difference in the contribution matrix.
    """
    # phi is p X T
    
    Y = phi.T @ A.T
    
    contributions = np.array([check_contribution_i(phi[i,:], A[:,i], Y) for i in range(phi.shape[0])])

    diff_mat = np.abs(contributions.reshape((1,-1)) - contributions.reshape((-1,1)))
    max_cont    = np.max(diff_mat )
    return max_cont    , np.unravel_index(np.argmax(diff_mat), diff_mat.shape)


def add_noisy_contribution(phi,A, max_cont_thres = 10):
    """
    Add noise to the neural maps A based on the similarity of contributions between temporal profiles phi and A.

    Parameters:
    - phi (numpy.ndarray or dict): The temporal profiles. Shape (p x T) or a dictionary of temporal profiles.
    - A (numpy.ndarray): The neural maps. Shape (N x p).
    - max_cont_thres (float): The maximum contribution difference threshold. If the maximum difference exceeds this threshold, noise is added. Default: 10.

    Returns:
    - numpy.ndarray: The updated neural maps A with added noise.
    """
    if isinstance(phi,dict):
        for key in phi.keys():
            print(key)
            max_cont,argmax = check_similar_contirubtion(phi[key], A)
            A = add_noisy_contribution(phi[key],A, max_cont_thres )
    else:
        max_cont,argmax = check_similar_contirubtion(phi, A)
        if max_cont > max_cont_thres:

            A[:, argmax[0]] += np.random.randn(A[:, argmax[0]].shape)
            A[:, argmax[1]] += np.random.randn(A[:, argmax[1]].shape)
    return A
    
def check_contribution_i(phi_i, A_i, Y):
    """
    Calculate the contribution difference between the reconstructed data Y and the product of phi_i and A_i.

    Parameters:
    - phi_i (numpy.ndarray): The temporal profile phi without the i-th component. Shape (p-1, T).
    - A_i (numpy.ndarray): The neural map A without the i-th component. Shape (N, p-1).
    - Y (numpy.ndarray): The reconstructed data. Shape (T, N).

    Returns:
    - float: The mean squared difference between the reconstructed data Y and the product of phi_i and A_i.
    """    
    # phi_i is phi without the i-th components. p-1 X T
    # A_i is A without the i-th component. N X p-1
    # Y is T x N
    
    return np.mean((Y -  phi_i.reshape((-1,1)) @ A_i.reshape((1,-1)))**2)   
    
    
def reorder_A(A):
    """
    Reorder the columns of the neural map A randomly.
    
    Parameters:
    - A (numpy.ndarray): The neural map. Shape (N, p).
    
    Returns:
    - numpy.ndarray: The neural map A with columns randomly reordered.
    """
    random_locs = [np.random.choice(np.arange(A.shape[0]), A.shape[0], replace = False).reshape((-1,1)) 
                   for _ in range(A.shape[1])]

    A = np.hstack([A[random_locs[i],i].reshape((-1,1)) for i in range(len(random_locs)) ])
    
    
    return A
    

def no_empties(A):
    """
    Handle empty rows in the neural map A by assigning non-zero random values.

    Parameters:
    - A (numpy.ndarray): The neural map. Shape (N, p).

    Returns:
    - numpy.ndarray: The modified neural map A.
    """
    empties = np.where(A.sum(1) == 0)[0]
    locs_empties = np.random.randint(0,A.shape[1],size= len(empties))
    for i, empty in enumerate( empties):
        A[empty, locs_empties[i]] = np.random.randn()
    return A
    
    
    
    
def create_grannet_synth_data(N, p, opts = opts,    labels = [1,2,3], opts_label_corr = {}, opts_temporal_corr = {}, opts_neural_corr = {}):
    """
    Create synthetic data for the Grannet model.

    Parameters:
    - N (int): Number of samples.
    - p (int): Number of neural maps.
    - opts (dict): Options for data generation. Default is opts.
    - labels (list): List of labels. Default is [1, 2, 3].
    - opts_label_corr (dict): Options for label correlation. Default is {}.
    - opts_temporal_corr (dict): Options for temporal correlation. Default is {}.
    - opts_neural_corr (dict): Options for neural map correlation. Default is {}.

    Returns:
    - numpy.ndarray: The neural map A. Shape (N, p).
    - numpy.ndarray: The temporal profiles phi. Shape (p, T).
    """    
    opts_temporal_corr = {**{ 'thres_corr_min' : 0.1, 'thres_corr_max' : 0.9}, **opts_temporal_corr}
    opts_neural_corr = {**{'threshold': {'min':0.1,'max':0.9}, 'type_return':'minmax'}, **opts_neural_corr}
    # init data
    A, phi = init_data(N, p, opts, labels )
    
    # define conditions to be met
    conds = {key: False for key in opts['conds']}
    
    # do not check labels correlations for lack of labels
    if len(labels) == 0 and 'corr_labels' not in opts.get('conds'):
        conds['corr_labels'] = False
    
    counter = 0
    phi_former = phi 
    A_former = A + 1
    while not phi_former == phi or not (A_former == A).all():
        A_former = A
        phi = phi_former
        
        print('checking corr with label')
        # if there are labels - check temporal
        if conds.get('corr_labels'):
            phi = check_corr_label(phi, labels, **opts_label_corr)
        
        print('checking corr between temporal')
        # check correlation between each pair of temporal profiles in c
        if 'corr_temps' in opts['conds']:
            phi =  update_corr_temporal(phi, **opts_temporal_corr)
        
        print('checking corr between neural maps')
        # check correlation between each pair of neural maps in A
        if 'orth_nets' in opts['conds']:
            A =  update_due_to_corr(A, **opts_neural_corr)
            # 
            
        print('checking similar contribution')
        if 'similar_cont' in opts['conds']:
            A = add_noisy_contribution(phi,A, opts['max_cont_thres'])
            
        print('sparsity')
        
        if (A_former != A).any():
            A = np.hstack([take_HT_col(A[:,i], opts['n_nonzeros'][i]).reshape((-1,1)) for i in range(p)     ])
        
        A = no_empties(A)
        print(counter)
        counter += 1
    return A, phi
            

        
"""
plotting
"""        
        
def add_labels(ax, xlabel='X', ylabel='Y', zlabel='', title='', xlim = None, ylim = None, zlim = None,xticklabels = np.array([None]),
               yticklabels = np.array([None] ), xticks = [], yticks = [], legend = [], ylabel_params = {},zlabel_params = {}, xlabel_params = {},  title_params = {}):
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
  
  

def apply_changes_to_A(A,type_change = 'miss', num_change = 4, labels = [1,2,3]):
    """
    Apply changes to the neural map A based on the specified type of change.

    Parameters:
    - A (numpy.ndarray): The neural map. Shape (N, p).
    - type_change (str): Type of change to apply. Default is 'miss'.
    - num_change (int): Number of changes to apply. Default is 4.
    - labels (list): List of labels. Default is [1, 2, 3].

    Returns:
    - dict: A dictionary of modified neural maps, with labels as keys.
    """    
    # remove neurons from each net of A
    A_dict = {}
    
    for label in labels:
        A_copy = A.copy()    
        for net in range(A.shape[1]):
            np.random.seed(label*net + net*net+label*(net+1))
            locs_not_zero = np.where(A[:,net] != 0)[0]
            print(locs_not_zero)
            to_null = np.random.choice(locs_not_zero , np.min([len(locs_not_zero), num_change]), replace = False)
            A_copy[to_null,net] = 0
        A_dict[label] = A_copy
    return A_dict
            
    
    
    
    

def plotting_results(A,phi, path_to_save = r'./', to_save = True, cmap = 'BrBG') :
    """
    Plot and save visualizations of the neural map A, temporal profiles phi, and histogram of the number of clusters for each neuron.

    Parameters:
    - A (numpy.ndarray): The neural map. Shape (N, p).
    - phi (dict): The temporal profiles. Dictionary of {condition: temporal profile}. Each temporal profile has shape (T, N).
    - path_to_save (str): Path to the directory where the figures will be saved.
    - to_save (bool): Flag indicating whether to save the figures or not. Default is True.
    - cmap (str): Color map for the heatmap visualization of A. Default is 'BrBG'.

    Returns:
    - None
    """    
    fig_A, ax_A = plt.subplots()
    sns.heatmap(A,  ax = ax_A, cmap = cmap, vmin = -3, vmax = 3)
    add_labels(ax_A, xlabel = 'clusters', ylabel = 'Neuron', title = 'A')
    if to_save:
        fig_A.savefig(path_to_save + os.sep + 'A.png' )
    
    fig_phi, ax_phi = plt.subplots(len(phi), 1, figsize = (len(phi)*5, 5))
    [ax_phi[i].plot(val.T) for i, (key,val) in enumerate(phi.items())]
    [add_labels(ax_phi[i], xlabel = 'Time', ylabel = 'Temporal Value', title = '$\Phi$, condition %d'%(i+1)) for i in range(len(ax_phi))]
    [ax_phi_spec.set_xlim(left = 0) for ax_phi_spec in ax_phi]
    fig_phi.tight_layout()
    if to_save:
        fig_phi.savefig(path_to_save + os.sep + 'phi.png' )
    
    # number of nets for a neuron
    vals = np.sum(A != 0, axis = 1)
    fig, ax = plt.subplots()
    ax.hist(vals)
    add_labels(ax, title = 'histogram # of clusters for a neuron', xlabel = '# of clusters', ylabel = 'count')
    if to_save:
        fig.savefig(path_to_save + os.sep + 'A_hist.png' )


"""
running explanation

data to save should be neuron X time X trial
"""   
to_run = False
make_multi_trial =  False #True


cmap = 'BrBG'
path_to_save = r'./'
if to_run:
    data_name = 'synth_grannet' 
    A,phi = create_grannet_synth_data(100,10)
    plotting_results(A,phi)
 
    """
    create labels
    """
    labels = np.array(list(phi.keys()))
    labels_name = create_data_name(data_name = data_name,  type_name = 'labels')
    A_dict = apply_changes_to_A(A,type_change = 'miss', num_change = 5, labels = labels)
    
    
    Y = {key:A_dict[key] @ phi_spec for key, phi_spec in phi.items()}
    
    data_name_save = create_data_name(data_name = data_name,  type_name = 'data')
    

    """
    stack data according to labels 
    """
    Y_stacked = np.dstack([Y[lab] for lab in labels])
    
    fig,ax = plt.subplots(1,3)
    [sns.heatmap(Y_spec, ax = ax[i], cmap = cmap, vmin = -5, vmax = 5) 
     for i, Y_spec in enumerate(Y.values())]
    [add_labels(ax_A, xlabel = 'time', ylabel = 'Neuron', title = 'Y%d'%(i+1)) for i, ax_A  in enumerate(ax)]
  
    fig.savefig(path_to_save + os.sep + 'data.png' )
    
    fig,ax = plt.subplots(10,10, sharex = True,)
    ax_or = ax.copy()
    ax = ax.flatten()
    colors = ['r','g','b']
    [[ax_s.plot(Y_spec[ax_s_count,:], color = colors[i], lw = 2) for ax_s_count,ax_s in enumerate(ax)]  
     for i, Y_spec in enumerate(Y.values())]

    [ax_s.set_xlabel('time') for ax_s in ax_or[-1,:]]
    fig.savefig(path_to_save + os.sep + 'data_plot.png' )
    fig,ax = plt.subplots(); [ax.plot([],[],color = colors[i], lw = 5, label = 'cond %d'%(i+1)) 
                              for i, ax_spec in enumerate(colors)]; ax.legend(prop = {'size':20})
    

    """
    after missing
    """

    fig,ax = plt.subplots(1,3)
    [sns.heatmap(A_spec, ax = ax[i], cmap = cmap, vmin = -3, vmax = 3) for i, A_spec in enumerate(A_dict.values())]
    [add_labels(ax_A, xlabel = 'clusters', ylabel = 'Neuron', title = 'A%d'%(i+1)) for i, ax_A  in enumerate(ax)]
    fig.savefig(path_to_save + os.sep + 'A_after_missing.png' )
    
    
    
    fig, ax = plt.subplots(1,3,  figsize = (20,5), sharex = True)
    [ax[i].hist(np.sum(A_spec != 0, axis = 1) , align='mid') for i, A_spec in enumerate(A_dict.values())]
    fig.suptitle( 'histogram # of clusters for a neuron')
    [add_labels(ax_s, title = 'A%d'%(ax_count+1), xlabel = '# of clusters', ylabel = 'count') 
     for ax_count, ax_s in enumerate(ax)]
    fig.savefig(path_to_save + os.sep + 'A_hist_specific.png' )


    """
    save data full
    """
    np.save('grannet_synth_results_march_2023.npy', {'A':A_dict, 'phi':phi, 'Y':Y})
    np.save(labels_name, labels)  
    np.save(data_name_save, Y_stacked)



num_trials = 10
std_A = 0.05
std_phi = 0.5
if make_multi_trial:
    data_name = 'synth_trials_grannet'
    d_full = np.load(r'grannet_synth_results_march_2023.npy', allow_pickle=True).item()
    labels = np.repeat(list(d_full['A'].keys()),num_trials)
    d_full['labels'] = labels
    A_real = d_full['A']
    phi_real = d_full['phi']
    Y_real = d_full['Y']
    
    A_real_mat = np.dstack([A_real[key] for key in range(1,4)])      # (N , T, k)
    A_real_mat = np.repeat(A_real_mat, num_trials, axis  = 2)
    phi_real_mat = np.dstack([phi_real[key] for key in range(1,4)]).transpose((1,0,2))  # (T,p, k)
    phi_real_mat = np.repeat(phi_real_mat, num_trials, axis  = 2)
    Y_real_mat = np.dstack([Y_real[key] for key in range(1,4)])      # (N, T, k)
    Y_real_mat = np.repeat(Y_real_mat, num_trials, axis  = 2)
    
    A_real_noisy = A_real_mat + np.random.randn(*A_real_mat.shape)*std_A
    phi_real_noisy = phi_real_mat + np.random.randn(*phi_real_mat.shape)*std_phi
    Y_reco = np.dstack([ A_real_noisy[:,:,k] @ phi_real_noisy[:,:,k].T  for k in range(A_real_noisy.shape[2])])
    
    np.save('grannet_synth_trials_results_march_2023.npy', {'A':A_real_noisy, 'phi':phi_real_noisy, 'Y_no_noise':Y_real_mat, 
                                                            'Y':Y_reco, 
                                                            'labels':labels, 'num_trials':num_trials})
    
    labels_name = create_data_name(data_name = data_name,  type_name = 'labels')
    data_name_save = create_data_name(data_name = data_name,  type_name = 'data')
    np.save(labels_name, labels)  
    np.save(data_name_save, Y_reco)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
