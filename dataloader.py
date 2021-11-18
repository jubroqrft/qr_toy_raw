from torch.utils.data import DataLoader
import torch
import numpy as np 
from util import scaler
from labelling import XYGeneration

def xy_generation(price_df, mu_df, sigma_df, dfy, args):

    #### scaling ####
    price_df_mm = scaler(price_df)
    mu_df_mm = scaler(mu_df)
    sigma_df_mm = scaler(sigma_df)

    #### Dataset Generation & Labelling 
    # Loading generator
    generator = XYGeneration(dfy, args.tr_past_seq, args.rf)

    # Y labelling 
    Y, Y_date, Y_tot_date = generator.Y_generation(args.pr1yr)

    # feature slicing & date slicing 
    x_p, X_date = generator.X_generation(price_df_mm, args.in_past_seq)
    x_m, _ = generator.X_generation(mu_df_mm, args.in_past_seq)
    x_s, _ = generator.X_generation(sigma_df_mm, args.in_past_seq)

    # features concatenating
    X = []
    for i in range(len(X_date)):
        x_p_arr = np.expand_dims(x_p[i], -1)
        x_m_arr = np.expand_dims(x_m[i], -1)
        x_s_arr = np.expand_dims(x_s[i], -1)
        X_arr = np.concatenate([x_p_arr, x_m_arr, x_s_arr], axis = -1)

        X.append(X_arr)

    num_samples = len(X)
#     num_train = round(len( X)*0.7) + one_step_ahead
    num_train = round(len(X)*0.7)
    num_step = num_samples - num_train

    return X, Y, X_date, Y_date, Y_tot_date, num_train, num_step


############# Dataloader ###########

def dataloader(X, Y, X_date, Y_date, Y_tot_date, num_train, step):
    num_train = num_train + step
    ## train & test split
    x_train, y_train = X[:num_train],Y[:num_train]
    x_test , y_test = X[num_train : num_train+1],Y[num_train : num_train+1]

    ## iterator
    train_iter = ([(torch.from_numpy(x).float(),torch.from_numpy(y).float()) for x,y in zip(x_train, y_train)])
    test_iter = ([(torch.from_numpy(x).float(),torch.from_numpy(y).float()) for x,y in zip(x_test, y_test)])

    ## dataloader
    train_dataloader = DataLoader(train_iter, batch_size=12, shuffle=True, drop_last = True)
    test_dataloader = DataLoader(test_iter, batch_size=1, shuffle=True, drop_last = True)

    ## date slicing
    # 미래 한달 
    test_date_x_m, test_date_y_m = X_date[num_train:num_train+1],Y_date[num_train:num_train+1]
    test_date_y_total = Y_tot_date[num_train:num_train+1] # labelling 계산할 때 씌인 기간 


    return train_dataloader, test_dataloader, test_date_x_m, test_date_y_m, test_date_y_total