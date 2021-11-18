import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from tqdm import tqdm
from dtaidistance import dtw
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


def pd_setting(dataframe):
    dataframe.rename(columns = {'Unnamed: 0':'Date'}, inplace = True)
    dataframe['Date'] = dataframe.Date.apply(lambda x : datetime.datetime.strptime(x, '%Y-%m-%d') )
    dataframe = dataframe.set_index(['Date'])

    return dataframe


def adj_generate(df_corr, min_threshold = 0.9, max_threshold = 1):
    '''
    input : df_corr(dataframe representing correlation)
    output : dataframe with 0, 1 (connectivity)
    '''
    df_adj = df_corr.applymap(lambda x : 1 if (abs(x)>min_threshold) & (abs(x)<max_threshold) else 0)
    return df_adj
    

def return_volatility(df_bench):
    '''
    return & volatility
    '''  
    logret = np.log(df_bench/df_bench.shift(1)).dropna()
    T = logret.shape[0]

    sigma = logret.cov() * T # annualize sigma (check again)
    mu = logret.mean()*T

    return mu, sigma


def dtw_sim_matrix(df_z):
    '''
    input : z_normalized df(concated ver)
    return : similarity value (array form) & heatmap 
    '''
    col_name = list(df_z.columns)
    num_col = len(col_name)

    dtw_pair = {}
    before_col = []

    # df_z : scaled df
    for col1 in tqdm(col_name):
        ind1 = col_name.index(col1)
        s1 = df_z[col1].to_numpy()
        before_col.append(col1)
        # only consider non-calcuated pairs
        col_name2 =  [item for item in col_name if item not in before_col] +([col1]) # 자기자신도 포함 

        for col2 in (col_name2):
            ind2 = col_name.index(col2)
            s2 = df_z[col2].to_numpy()
            # dtw calculation 
            sim = dtw.distance_fast(s1, s2)
            sim = round(sim, 2)
            dtw_pair[(ind1, ind2)] = sim

    sim_arr = np.zeros((num_col,num_col))
    
    for i in range(num_col):
        for j in range(i, num_col):
            key = (i,j)
            sim_value = dtw_pair[key]
            sim_arr[i][j] = sim_value
    # symmetric matrix generation 
    sim_arr_all = sim_arr + sim_arr.T 
    sim_df = pd.DataFrame(sim_arr_all, columns = col_name, index = col_name )
    # map setting
    plt.figure(figsize = (20,10))
    plt.rc('xtick', labelsize=20) 
    plt.rc('ytick', labelsize=20) 
    # heatmap 
    g = sns.heatmap(sim_df, square = True, linewidth= 2,  cmap="rocket_r")
    g.set_xticklabels(g.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.title("DTW similarity", fontsize =(20))
    plt.show()

    return sim_arr_all


def scaler(df):
    df_col_name = df.columns
    feat_date = list(df.index)

    min_max_scaler = MinMaxScaler()
    min_max_scaler.fit(df)
    df_mm = min_max_scaler.transform(df)
    df_mm = pd.DataFrame(df_mm, columns = df_col_name, index = feat_date)
    
    return df_mm
    