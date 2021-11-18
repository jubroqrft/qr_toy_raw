import pandas as pd
import numpy as np 
import datetime 
import pickle 
import random





df = pd.read_csv("./data/Toy_data.csv")
df.rename(columns = {'Unnamed: 0':'Date'}, inplace = True)
df['Date'] = df.Date.apply(lambda x : datetime.datetime.strptime(x, '%Y-%m-%d') )
df = df.set_index(['Date'])

df_tar = pd.read_csv("./data/Toy_data_target.csv")
df_tar['Date'] = df_tar.Date.apply(lambda x : datetime.datetime.strptime(x, '%Y-%m-%d') )
df_tar = df_tar.set_index(['Date'])

### X input 의 시작점과 동일하게 
df_tar = df_tar.loc["1990-01-01":]

### X input
# Nan value 많은 column 제거 
df = df.drop(columns = ['MXUSMMT Index'], axis = 1)
# 나머지 채워넣기 
df = df.fillna(method = 'bfill')


### X Y concatenation 

dff = copy.deepcopy(df_tar) # target copied 
dff_col_name = list(dff.columns)
dff_index = list(dff.index)
# Y asset 이름 바꿔주기
dff_col_name_y = [name+'_Y' for name in dff_col_name]    
dfy = pd.DataFrame(dff.values, columns = dff_col_name_y, index = dff_index)

### X, Y 합쳐주기 
dff = pd.concat([df,dfy], axis = 1)

# column drop 
dff = dff.drop(columns = ['.TED G Index', 'USYC2Y10 Index'], axis = 1)

### target 의 EMB 데이터 Nan 값 추가해주기 
emb = dff[['EMB_Y']].dropna(axis = 0)
emb.index.name = 'Date'
# 뒤집어서 구한다. 
x = np.arange(emb.index.size)
emb_rev = emb.values[::-1]

# split last value to 0 value
def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

split_last = list(split(range(int(em_ls[-1])),804))
split_mean = [np.mean(x) for x in split_last] # split 된 범위의 평균값으로 취해줌

prediction = [em_ls[-1]-x for x in split_mean] # 하나씩 빼가면서 추가
emb_exp = em_ls + prediction 
emb_exp2 = emb_exp[::-1]

### 새로운 데이터 프레임 생성
EMB_Y_exp = pd.DataFrame(np.array(emb_exp2), columns = emb.columns, index = dff.index)
asdf = dff.drop('EMB_Y', axis = 1)
dff = asdf.join(EMB_Y_exp)

