import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
import argparse
import os
import time 
from tqdm import tqdm
from util import *
from dataloader import *
from LSTMmodel import *

# 500 mins for 101 cases

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cuda:1',help='')
parser.add_argument('--in_past_seq', type = int, default=240, help='number of past sequence for x input')
parser.add_argument('--tr_past_seq', type = int, default=252, help='number of past sequence for y label(ex, 253 one year trading days)')
parser.add_argument('--rf',type=int,default=0.02,help='risk free rate')
parser.add_argument('--pr1yr', type=bool, default=False, help = 'whether pr1yr for labelling')
parser.add_argument("--test_return_month", type = bool, default = True, help = 'whether to calculate future return and volatility with only one future month')
parser.add_argument('--num_epochs',type=int,default=50,help='')
parser.add_argument('--dropout',type=int,default=0.3,help='')
parser.add_argument('--learning_rate',type=int,default=0.001,help='')
parser.add_argument('--wdecay',type=int,default=0.0001,help='')
parser.add_argument('--gcn_bool',type=bool,default=True,help='')
parser.add_argument('--addaptadj',type=bool,default=True,help='')
parser.add_argument('--aptinit',type=int,default=None,help='')
parser.add_argument('--in_dim',type=int,default=3,help='')
parser.add_argument('--out_dim',type=int,default=1,help='')
parser.add_argument('--out_dim2',type=int,default=9,help='')
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--kernel_size',type=int,default=40,help='')
parser.add_argument('--num_block',type=int,default=4,help='')
parser.add_argument('--num_layer',type=int,default=2,help='')

args = parser.parse_args()


# output directory 
output_dir = "./RESULT/LSTM_{}_seq".format(args.in_past_seq)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


t1 = time.time()
################################ shared process ########################################
#### feature loading ####
price_df = pd_setting(pd.read_csv("./data/features/price_df.csv"))
mu_df = pd_setting(pd.read_csv("./data/features/mu_df.csv"))
sigma_df = pd_setting(pd.read_csv("./data/features/sigma_df.csv"))
    # variables
feat_date = price_df.index
dff_col_name = list(price_df.columns)
num_nodes = len(dff_col_name)


#### Y Loading ####
dff = pd.read_csv("./data/df_XY.csv")
dff = pd_setting(dff)
dfy = dff.loc[feat_date]
# extracting Y columns
Y_col = [x for x in dfy.columns if x[-2:]=='_Y']
dfy = dfy[Y_col]

X, Y, X_date, Y_date, Y_tot_date, num_train, num_step = xy_generation(price_df, mu_df, sigma_df, dfy, args)

#########################################################################################
for step in tqdm(range(0, num_step)):
    train_dataloader, test_dataloader, test_date_x_m, test_date_y_m, test_date_y_total = dataloader(X, Y, X_date, Y_date, Y_tot_date, num_train, step)
    
    ################### DataLoader #################
    print ("**************************************************")
    print ("DataLoader with step : {}".format(step))

    ################### Model ######################
    num_nodes = 27
    past_seq = args.in_past_seq
    Y_asset_num = 9
    dim_num = 3
    num_layers = 1
    learning_rate = args.learning_rate
    wdecay = args.wdecay
    
    # model 
    model = LSTM_Model(num_nodes, past_seq, Y_asset_num, dim_num, num_layers, args.device)

   

    model.to(args.device) # model loading

    criterion = torch.nn.L1Loss() # MAE  
    optimizer = torch.optim.Adam(model.parameters(), lr = args.wdecay, weight_decay = args.wdecay)

    #################### Training ###################


    loss_train = []

    for epoch in range(1, args.num_epochs+1):
        model.train()

        epoch_train_loss = [] # loss average for one epoch
        epoch_valid_loss = []
    ##### training 
        for data in train_dataloader:
            optimizer.zero_grad()

            train_X, train_Y = data

            train_X = torch.Tensor(train_X).to(args.device)
            train_Y = torch.Tensor(train_Y).to(args.device)

            train_out = model(train_X)
            loss = criterion(train_out,train_Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss.append(loss.item())

        loss_train.append(np.mean(epoch_train_loss))

        if epoch%10 == 0:
            print ("Epoch {}| tarin loss : {}".format(epoch,np.mean(epoch_train_loss)))

    ############## TESTING ####################
    model.eval()
    test_loss = []
    ##### TESTING ##############
    for data in test_dataloader:
        test_X, test_Y = data
        test_X = torch.Tensor(test_X).to(args.device)
        test_Y = torch.Tensor(test_Y).to(args.device)

        test_out = model(test_X)
        test_out =torch.unsqueeze(test_out, 0)

        loss = criterion(test_out, test_Y)

        test_loss.append(loss.item())

    print ("{} fold test loss(MAE): {}".format(step, test_loss[0]))

#     print ("test_ out shape :", test_out.shape)
    ### 라벨링 할 때 쓰였던 동일한 기간에 대한 수익률과 volatility 구한다. 
    # price slicing
    if args.test_return_month == True:
        test_price = dfy.loc[test_date_y_m[0][0]:test_date_y_m[0][-1]]
    else:   
        test_price = dfy.loc[test_date_y_total[0][0]:test_date_y_total[0][-1]]
    # bench mu, volatility calculation 
    test_mu, test_sigma = return_volatility(test_price)
    # weight setting
    pred_w = test_out.detach().cpu().numpy()
#     print (pred_w)
    true_w = test_Y.detach().cpu().numpy()
    # mu
    pred_mu = pred_w.dot(test_mu.values)
    true_mu = true_w.dot(test_mu.values)
    # sigma
    pred_sigma = np.sqrt(pred_w.dot(test_sigma.values).dot(pred_w.T))
    true_sigma = np.sqrt(true_w.dot(test_sigma.values).dot(true_w.T))

    ############### Saving as ond DataFrame ###############
    label_begin_date, label_end_date = list(test_price.index)[0], list(test_price.index)[-1]

    fold_dict = {}
    
    fold_dict['Date']= [([[label_begin_date, label_end_date]])]
    fold_dict['X_date'] = [([[test_date_x_m[0], test_date_x_m[-1]]])]
#     print(pred_w)
    fold_dict['pred_w']= [list(pred_w)]
    fold_dict['pred_mu']= [pred_mu]
#     fold_dict['pred_sigma']=pred_sigma
    fold_dict['true_w']=[list(true_w)]
    fold_dict['true_mu']= [true_mu]
#     fold_dict['true_sigma']=true_sigma
    fold_dict['test_loss(%)']=round(test_loss[0], 4)*100

    fold_result_df = pd.DataFrame.from_dict(fold_dict)
    
    with open(os.path.join(output_dir, "result_{}.pickle".format(step)), 'wb') as f:
        pickle.dump(fold_result_df, f)

    print ("Saving Result...")
    # GPU cache 삭제 
    torch.cuda.empty_cache()