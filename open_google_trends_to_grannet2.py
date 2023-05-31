# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 17:52:00 2022

@author: ---
"""

"""
Define Parameters
"""
global path_csvs
path_csvs = r'./' # WHERE THE FILES ARE STORED?
name_csv = 'ai'
last_date = '2022-10'

"""
Imports
"""
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import glob
from main_functions_graft_lab2 import *

"""
Functions
"""

def create_dict_date_to_number(dates_list):
    """
    Parameters
    ----------
    dates_list : list of dates (i.e., ['2004-03', '2004-04','2004-07', '2004-08'])

    Returns
    -------
    dict_dates2numbers : dict with dates as keys, numbers as values
    dict_numbers2dates : dict with dates as values, numbers as keys
    """
    dict_dates2numbers = {date: num for num,date in enumerate(dates_list)}
    dict_numbers2dates = {num:date for num,date in enumerate(dates_list)}
    return dict_dates2numbers, dict_numbers2dates

def from_df_to_dateslist(df, last_date = '2022-10', start_date  = '2011-01'):

    dates_vals = df.index.values.astype(str)
    if dates_vals[0].lower() ==  'month':
        dates_vals = dates_vales[1:]
        df = df.iloc[1:,:]
    where_start = np.where(dates_vals == start_date)[0][0]  
    where_stop = np.where(dates_vals == last_date)[0][0] + 1
    df = df.iloc[where_start:where_stop,:]
    return df, dates_vals    
    
def put_title_for_dfs(df, remove_loc = True)    :
    pot_cols = df.iloc[0].values
    if remove_loc:
        pot_cols = [col.split(':')[0] for col in pot_cols]
    df.columns = pot_cols
    df = df.iloc[1:,:]
    return df

def read_dfs_to_one_df(path = path_csvs, last_date = '2022-10', start_date = '2011-01'):
    files_names = os.listdir(path)
    ex_index = put_title_for_dfs(pd.read_csv(path + os.sep + files_names[0])).index.values
    list_dfs = [put_title_for_dfs(pd.read_csv(path + os.sep + file_name)).loc[ex_index,:] for file_name in files_names]
    dfs_concat = pd.concat(list_dfs, axis = 1)
    dfs_concat, dates_vals   = from_df_to_dateslist(dfs_concat, last_date, start_date)
    dict_dates2numbers, dict_numbers2dates = create_dict_date_to_number(dates_vals)
    return dfs_concat, dict_dates2numbers, dict_numbers2dates


"""
functions march 
"""
import os
def read_multi_csv_single(path = '', file_name = '', full_path = '',
                          terms_to_remove = ['AC','sun','moon','voter','candies','lab', 'fireworks']):
    if len(full_path) == 0:
        if len(path) == 0:
            path = r'E:\CODES FROM GITHUB\GraFT_Python\datasets\GOOGLE_TRENDS\new_by_state'
        if len(file_name) == 0:
            file_name = r'WA8.csv'
        if not file_name.endswith('.csv'):
            file_name = file_name + '.csv'
        full_path = path + os.sep + file_name
    df = pd.read_csv(full_path ,  skipinitialspace=True, skiprows=2)
    """
    fix hebrew columns
    """
    df_columns = np.array(['month'] + [col.split(':')[0] for col in df.columns[1:]])
    df.columns = df_columns
    
    df.set_index('month', inplace = True)
    df = df.replace(sss,1)
    df = df.astype(np.int)
    df ,_ = from_df_to_dateslist(df, last_date = '2022-10', start_date  = '2011-01')
    return df
    
sss = np.load('replace_small.npy',allow_pickle=True).item()
def read_multi_csv_all_path(path = '', sss = sss):
    if len(path) == 0:
        path = r'./'
    files = glob.glob(path + os.sep + '**.csv')
    df_dict = {}
    """
    run over files
    """
    for file in files:
        state = file.split(os.sep)[-1][:2]
        
        if state != 'AC':
            df = read_multi_csv_single(full_path = file)
            """
            min max normalize
            """
            # """
            # MARCH STYLE
            # """
            # df = (df- df.min(0))*100 / np.array(df.max(0)- df.min(0)).reshape((1,-1))
            
            """
            may style
            """
            wind = 14
            vals = df.values
            nani = np.nan*np.zeros((df.shape[0], df.shape[1], df.shape[0]))
            for i in np.arange(df.shape[0]):
                vals_cur = vals[np.max([0, i - int(wind/2)]): np.min([df.shape[0], i + int(wind/2)]),:]
                vals_cur = (vals_cur - np.min(vals_cur, 0)) / (vals_cur.max(axis = 0) - np.min(vals_cur, axis = 0))

                nani[np.max([0, i - int(wind/2)]): np.min([df.shape[0], i + int(wind/2)]), :,i]  = vals_cur                        
            df = pd.DataFrame(np.nanmean(nani, axis = 2), index = list(df.index), columns = df.columns)
            df.fillna(0)

            df = (df- df.min(0))*100 / np.array(df.max(0)- df.min(0)).reshape((1,-1))
            df.fillna(0)
            if state not in df_dict.keys():
                df_dict[state] = pd.DataFrame()
            df_dict[state] = pd.concat([df_dict[state], df],axis = 1)
    df_list, df_dict , df_label = vstack_dfs(df_dict, terms_to_remove=terms_to_remove)
    return df_dict, df_list, df_label
   
    
    
def vstack_dfs(df_dict,terms_to_remove = []):
    df_list = []
    df_label = []
    cols = []
    for state, df in df_dict.items():
        df_label = df_label + [state]*df.shape[0]
        df = df.loc[:,~df.columns.duplicated()].copy()
        df_list.append(df)
        df = df.fillna(0)
        cols.extend([col.lower() for col in list(df.columns)])

    
    df_list  = pd.concat(df_list, axis = 0)
 
    df_label = np.array(df_label) # these are the states
    inds_to_keep = df_list.isna().any() == False

    inds_to_keep = inds_to_keep.values 
    df_list = df_list.iloc[:,inds_to_keep]
    terms = np.array(list(df_list.columns)).astype(str)
    terms = np.array([term.capitalize() for term in terms])
    df_list.columns = terms 
    terms_to_remove = np.array([term.capitalize() for term in terms_to_remove])
    result = np.setdiff1d(terms, np.array(terms_to_remove))
    df_list = df_list.loc[:,result]
    df_list = df_list.sort_index(axis=1)
    df_label = np.array(df_label)
    df_label[df_label == 'LU'] = 'LA'
    df_dict = {state:df_list.iloc[df_label == state,:] for state in np.unique(df_label)}
    
    return df_list, df_dict, df_label
    
    
  
  
def add_labels(ax, xlabel='X', ylabel='Y', zlabel='', title='', xlim = None, ylim = None, zlim = None,xticklabels = np.array([None]),
               yticklabels = np.array([None] ), xticks = [], yticks = [], legend = [], 
               ylabel_params = {'fontsize':19},zlabel_params = {'fontsize':19}, xlabel_params = {'fontsize':19},  title_params = {'fontsize':19}):
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

    
def plot_trends_march(df_list, df_dict, df_label, cmap = 'winter', 
                      colors = ['r','g','b','purple','orange','brown','black','cyan'])   :
    
    
    terms = list(df_list.columns)
    
    plt.figure(); sns.heatmap(df_list, cmap = cmap)
    plt.tight_layout()
    
    
    fig, ax = plt.subplots(1,len(df_dict), sharex = True, sharey = True)
    states = list(df_dict.keys())
    [sns.heatmap(df_dict[lab], ax = ax[i], vmin = 0, vmax = 100) for i,lab in enumerate(states)]
    [add_labels(ax_spec, title = states[i], xlabel = 'term', ylabel = 'time', zlabel = '') for i,ax_spec in enumerate(ax)]
    
    fig.tight_layout() 
    
    fig, ax = plt.subplots(9,9, sharex = True)
    ax = ax.flatten()
    [[ax[term_num].plot(df_dict[label].iloc[:,term_num].values, color = colors[label_num], lw = 2) for term_num, term in enumerate(terms)  ]    
     for label_num, label in enumerate(states)]
    [add_labels(ax[term_num], title = term, xlabel = '', ylabel ='', zlabel ='') for term_num, term in enumerate(terms)  ]
    [ax[-1].plot([],[],color = colors[state_num], label = state, lw = 10) for state_num,state in enumerate(states)]
    ax[-1].legend(prop = {'size':10})
    remove_edges(ax[-1])
    fig.tight_layout()
    
    
    """
    plot stacked switched
    """
    
    res_list2, states_full2, terms2 = make_switch_state_data_for_plot(df_dict)
    fig, axs = plt.subplots(1,2,figsize = (20,8) ,
                            gridspec_kw={'width_ratios': [res_list2.shape[1],len(np.unique(states_full2))*4 ]},
                           sharey = True)
    terms2 = np.array([term.capitalize() for term in terms2])
    terms = np.array([term.capitalize() for term in terms])
    ax = axs[0]
    sns.heatmap(res_list2, ax = ax, cmap = 'Greys', vmin = 0, vmax = 100, cbar = False)
    
    # Set xticks and labels at bottom
    ax.set_xticks(np.arange(0,res_list2.shape[1], len(np.unique(states_full2))) + len(np.unique(states_full2))/2 )
    ax.set_xticklabels(terms, fontsize = 20, fontweight = 'bold')
    
    # Set xticks and labels at top
    ax2 = ax.twiny()
    ax2.set_xticks(np.arange(res_list2.shape[1]) )
    ax2.set_xticklabels(states_full2, rotation = 90 , fontsize = 3)
    
    ax2.set_yticks([])
    ax.set_yticks([0.5, res_list2.shape[0]-0.5])
    ax.set_yticklabels(['2004', ' 2022'], fontsize = 20, fontweight = 'bold')    


    ax3 = axs[-1]
    
    sns.heatmap(res_list2[:,len(np.unique(states_full2))*14:len(np.unique(states_full2))*15], 
                ax = ax3, cmap = 'Greys', vmin = 0, vmax = 100, cbar = True)
    ax3.set_xticks(np.arange(0,len(np.unique(states_full2)))+ 0.5)
    ax3.set_xticklabels(np.unique(states_full2), fontsize = 10, fontweight = 'bold', rotation = 90)
    ax3.xaxis.set_label_position('bottom')
    ax3.xaxis.set_ticks_position('top')
    
    ax3.set_xlabel(terms2[len(np.unique(states_full2))*14], fontsize = 20, fontweight = 'bold')
 
    ax2.set_yticks([])
    ax.set_yticks([0.5, res_list2.shape[0]-0.5])
    ax.set_yticklabels(['2004', ' 2022'], fontsize = 20, fontweight = 'bold')    
    ax.set_ylabel('Time', fontsize = 20, fontweight = 'bold')
    ax.set_xlabel('Term', fontsize = 20, fontweight = 'bold')
    fig.tight_layout()
    create_cbar(res_list2, '', w_cmap=False, cmap = cmap, data_it = True, add_yticklabels=False)
    
    

    
def create_cbar(color_matrix, label_basic = 'k', alpha = 0.99, to_plot = True, inv = False, rot = 0, w_cmap = False, cmap = '',
                add_yticklabels = False, data_it = True):
    if w_cmap and len(cmap) == 0:
        if not inv:
            cmap = ListedColormap(color_matrix.T)
        else:
            cmap = ListedColormap(color_matrix.T[::-1,:])
    elif len(cmap) == 0:
        cmap = 'Blues'
    if to_plot:
        # Create a figure with a colorbar
        fig, ax = plt.subplots(figsize = (0.5,5))
        
        if data_it:
            list_min_max = [color_matrix.min(), color_matrix.max()]
            data_min = np.max(np.abs(list_min_max))*(-1)

            data_max = np.max(np.abs(list_min_max))
            im =  ax.scatter(np.linspace(data_min,data_max,3), 
                           np.linspace(data_min,data_max,3),
                            c = np.linspace(data_min,data_max,3), cmap = cmap)
          

        else:
            im = ax.scatter(np.arange(color_matrix.shape[1]), np.arange(color_matrix.shape[1]), 
                        c =  np.linspace(0,1, color_matrix.shape[1]), cmap  = cmap , alpha = alpha)
       
        len_each =  1/color_matrix.shape[1]*0.5
        
        if add_yticklabels:
            cbar = plt.colorbar(im, ax=ax, ticks=np.arange(color_matrix.shape[1]), cax = ax)
            cbar.set_ticks(np.linspace(len_each,1-len_each,color_matrix.shape[1]) )
            if not inv:
                cbar.ax.set_yticklabels(['$%s_{%d}$'%(label_basic,i) for i in np.arange(1, color_matrix.shape[1] +1)], 
                                    fontsize = 20, rotation = rot + 90)
            else:
                cbar.ax.set_yticklabels(['$%s_{%d}$'%(label_basic,i) for i in np.arange(1, color_matrix.shape[1] +1)[::-1]], 
                                    fontsize = 20, rotation = rot)
        else:
            cbar = fig.colorbar(im,  cax = ax)

        fig.tight_layout()

        return ax, cbar, cmap
    return 0,0, cmap

    
def make_switch_state_data_for_plot(df_dict):
    """
    Takes in a dictionary of dataframes where each dataframe corresponds to a particular switch state.
    The dataframes have rows indexed by time and columns indexed by query names. This function reshapes 
    the dataframes to a 2D numpy array where each row corresponds to a query and each column corresponds 
    to a timepoint, such that the data can be used to create a heatmap plot. The function returns the 
    reshaped numpy array, a list of switch states (in the same order as the columns of the array), and a 
    list of query names (in the same order as the rows of the array).
    
    Parameters:
    -----------
    df_dict : dict
        A dictionary of pandas dataframes. Each dataframe corresponds to a particular switch state, and 
        has rows indexed by time and columns indexed by query names.
    
    Returns:
    --------
    """
    res_list = []
    terms =  np.array(list(df_dict['CA'].columns)).astype(str)
    
    states_short = list(df_dict.keys())
    terms_full = np.repeat(terms, len(states_short))
    states_full = np.tile(states_short,len(terms))
    for query_num, query in enumerate(terms):
        for state in states_short :
            val = df_dict[state]
        
            vals = val.loc[:,query].values
            res_list.append(vals.reshape((-1,1)))
            
        
    return np.hstack(res_list), states_full, terms_full
            
            
            
            
      
    

"""
load_csv
"""
terms_to_remove = []
to_run = False
to_run_march = False
to_run_may = True
if to_run:
    plt.close('all')

    plt.figure()
    dfs_concat, dict_dates2numbers, dict_numbers2dates = read_dfs_to_one_df()
    
    dfs_concat[dfs_concat  == '<1'] = 1
    dfs_concat = dfs_concat.astype(np.float)
    
    dfs_concat_integers = dfs_concat.copy()

    quantile_val = dfs_concat.quantile(0.8,axis = 0).values
    for col_i, col in enumerate(dfs_concat.columns):
        dfs_concat.loc[(dfs_concat[col] > quantile_val[col_i]).values, col] = quantile_val[col_i]

    dfs_concat.columns = [col.lower() for col in dfs_concat.columns]
    dfs_concat = dfs_concat / dfs_concat.sum(0).values.reshape((1,-1))

    plt.figure()
    sns.heatmap(dfs_concat, robust = True, xticklabels = True)
    plt.figure()
    ClusterGrid = sns.clustermap(dfs_concat, robust = True, xticklabels = True)
    plt.figure()
    reordered = ClusterGrid.dendrogram_col.reordered_ind
    re_dfs = dfs_concat.iloc[:,reordered]
    sns.heatmap(re_dfs, robust = True, xticklabels = True)
    
    
    dfs_array = dfs_concat_integers.T.values
    data_name = create_data_name(data_name = 'trends')
    np.save(data_name,dfs_array)
    mapping_trends = {i:col for i,col in enumerate(dfs_concat.columns)}
    np.save('mapping_trends.npy',mapping_trends)
    
    

if to_run_march:    
    data_name = 'trends_grannet' 
    df_dict, df_list, df_label = read_multi_csv_all_path()
    plot_trends_march(df_list, df_dict, df_label)
    terms_list = df_list.index
 
    labels = list(df_dict.keys())
    labels_name = create_data_name(data_name = data_name,  type_name = 'labels')
    data_name_save = create_data_name(data_name = data_name,  type_name = 'data')
    dict_nums, list_nums = labels_to_nums(labels)
    np.save(labels_name, np.array(list_nums))  
    labels_full = create_data_name(data_name = data_name,  type_name = 'labels_full')
    np.save(labels_full, np.array(list_nums))
    Y_stacked = np.dstack([df_dict[key].values for key in labels])
    Y_stacked = Y_stacked.transpose([1,0,2])
    np.save(data_name_save, Y_stacked)
    terms = np.array(list(df_dict['CA'].columns)).astype(str)
    np.save('grannet_trends_results_march_2023.npy', {'df_dict':df_dict,
                                                      'df_list':df_list, 
                                                      'df_label':df_label,
                                                      'dict_nums':dict_nums,
                                                      'labels':labels,
                                                      'terms': terms})
    np.save('grannet_trends_for_jupyter_results_march_2023.npy', {
                                                      'labels':labels,
                                                      'terms': terms})
  
    
    
    
    
    
if to_run_may:    
    data_name = 'trends_gannet2' 
    df_dict, df_list, df_label = read_multi_csv_all_path()
    plot_trends_march(df_list, df_dict, df_label)
    terms_list = df_list.index
    
    labels = list(df_dict.keys())
    labels_name = create_data_name(data_name = data_name,  type_name = 'labels')
    data_name_save = create_data_name(data_name = data_name,  type_name = 'data')
    dict_nums, list_nums = labels_to_nums(labels)
    np.save(labels_name, np.array(list_nums))  
    labels_full = create_data_name(data_name = data_name,  type_name = 'labels_full')
    np.save(labels_full, np.array(list_nums))
    Y_stacked = np.dstack([df_dict[key].values for key in labels])
    Y_stacked = Y_stacked.transpose([1,0,2])
    np.save(data_name_save, Y_stacked)
    terms = np.array(list(df_dict['CA'].columns)).astype(str)
    np.save('grannet_trends2_results_march_2023.npy', {'df_dict':df_dict,
                                                      'df_list':df_list, 
                                                      'df_label':df_label,
                                                      'dict_nums':dict_nums,
                                                      'labels':labels,
                                                      'terms': terms})
    np.save('grannet_trends2_for_jupyter_results_march_2023.npy', {
                                                      'labels':labels,
                                                      'terms': terms})    