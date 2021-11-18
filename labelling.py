
import pandas as pd
import numpy as np 
import datetime 
import util

from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns

'''
X sequence 길이 조절해주는 함수 추가 

'''



class XYGeneration():
    '''
    Labelling
    
    df_input : X assets 
    df_target : Y assets
    tr_past_seq : past duration 
    rf : risk-free
    '''
    
    def __init__(self, df_target, tr_past_seq, rf):
        '''
        # in_past_seq < tr_past_seq (안 그러면, 처음 시작 점 엉킴 )
        df_input(X) starting datetime should be earlier or eqaul to that of 'df_target'. 
        df_input starting datetime can be later than that of df_target
        
        df_target(Y) starting datetime should be equal or greater than that of df_input
        '''
        self.df_target = df_target
        self.rf = rf
        self.tr_past_seq = tr_past_seq
        self.bench_dict = self.bench_dict_generation(self.df_target)
    
    def bench_dict_generation(self, df):
        '''
        Select first & last trading day of the month
        '''
        first_day = []
        last_day = []
        count = 0
        first_day.append(df.index[0])
        prev_month = df.index[0].month # 1
        for ind in range(len(df.index)):
            date = df.index[ind]
            if prev_month == 13:
                prev_month = 1
            month = date.month # current month

            if month != prev_month: # if current month != previous month
                count = 0
                if count == 0 : # first day of the month
                    first_day.append(date)
                    last_day.append(df.index[ind-1]) # last day of the previous month
                prev_month+=1
                count+=1

        # building benchmark date : fist date and last date of the month
        first_last_day_dict = {}
        key = [i for i in range(len(first_day))] # order number
        value = list(zip(first_day, last_day))
        bench_dict = dict(zip(key, value))

        return bench_dict
    
    def return_volatility(self, df_bench):
        '''
        return & volatility
        '''  
        logret = np.log(df_bench/df_bench.shift(1)).dropna()
        T = logret.shape[0]

        sigma = logret.cov() * T # annualize sigma (check again)
        mu = logret.mean()*T

        return mu, sigma

    def MVO_sharpe(self,mu,sigma):
        '''
        Try no package later
        '''
        
        mu, sigma = self.return_volatility(self.df_target)
        ef = EfficientFrontier(mu, sigma, weight_bounds = (0, 0.25))

        # Find the tangency portfolio
        ef.max_sharpe(risk_free_rate = self.rf) # risk free rate 추후 조정
        weights = ef.clean_weights() # rounding upto 5th decimal

        ret_tangent, std_tangent, _ = ef.portfolio_performance(risk_free_rate = self.rf, verbose = False)
        # LABEL 
        weights_label = np.array(list(weights.values()))

        return weights_label, ret_tangent, std_tangent

    def Y_generation(self, pr1yr=False):
        Y_label = []
        Y_fut_date = []
        Y_tot_date = []
        
        st_ind = (self.tr_past_seq//20) # 20 trading days in a month -> starting index for labelling 
        # date list 
        date_ls = list(self.df_target.index)

        for ind in range(st_ind, len(self.bench_dict)):
            
            # future
            future_first_t = self.bench_dict[ind][0] # current st date
            last_t = self.bench_dict[ind][1] # current last date
            
            if pr1yr == True:
                # pr1yr
                    # 12 개월을 보는데 바로 직전의 1달은 제외한다. 
                past_last_t = self.bench_dict[ind-1][1] # 전달의 마지막 날 까지 
            else:
                past_last_t = future_first_t
                
            begin_ind = date_ls.index(future_first_t) - self.tr_past_seq
            past_begin_t = date_ls[begin_ind]
            
            future = self.df_target[(self.df_target.index >= future_first_t) & (self.df_target.index <= last_t)] # future one month
            past = self.df_target[(self.df_target.index >= past_begin_t) & (self.df_target.index < past_last_t)] # past duration control
            
            # concatenating future and past
            df_bench = pd.concat([past, future]).drop_duplicates()
            mu, sigma = self.return_volatility(df_bench)
            # calculating weight maximizing sharpe ratio
            w, r, s = self.MVO_sharpe(mu,sigma)
   
            # appending
            Y_label.append(w)
            Y_fut_date.append((future_first_t, last_t))
            Y_tot_date.append(list(df_bench.index))
            
        return Y_label, Y_fut_date, Y_tot_date 
    
    def X_generation(self, df_input, in_past_seq):
        '''
        in_past_seq : number of days to consider for X sequence
        
        feature addition later after this function 
        
        '''
        date_ls = list(df_input.index)
     
        st_ind = self.tr_past_seq//20 # slicing beginning point

        X_input = []
        X_index = []
  
        for i in range(st_ind, len(self.bench_dict)):
            # last month last day 
            last_t = self.bench_dict[i-1][1]
            # first day index
            first_ind = date_ls.index(last_t) - in_past_seq
            first_t = date_ls[first_ind]
            df_t = df_input[(df_input.index <= last_t) & (df_input.index > first_t)]
            
            X_input.append(df_t.values)
            # date slicing
            X_index.append(list(df_t.index))
        
        return X_input, X_index
 
        
        

    
    
