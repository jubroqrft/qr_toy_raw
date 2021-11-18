import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable




class LSTM_Model(nn.Module):
    
    def __init__(self, X_assets_num, past_seq, Y_asset_num, dim_num, num_layers, device):
        super(LSTM_Model, self).__init__()
        self.device = device
        self.embedding_in_dim = dim_num
        self.num_layers = num_layers
        
        self.lstm_in_dim = X_assets_num
        self.lstm_out_dim = X_assets_num 
        
        self.fc_in_dim = past_seq
    
        self.fc2_out_dim = Y_asset_num
        self.embedding = nn.Conv2d(self.embedding_in_dim, 1, kernel_size = (1,1))
        self.lstm = nn.LSTM(self.lstm_in_dim, self.lstm_out_dim, num_layers = self.num_layers, batch_first = True)
        self.fc = nn.Conv2d(self.fc_in_dim, 1, kernel_size= (1,1))
        self.fc2 = nn.Conv2d(self.lstm_out_dim, self.fc2_out_dim, kernel_size = (1,1))
     
    def forward(self, x):
        batch_size = x.size(0)
        
        x = x.transpose(3,1)
        x = self.embedding(x)
        x = x.transpose(1,3)
        if x.shape[0] != 1:
            x = x.squeeze() # [b, L, N]
        else:
            x = torch.squeeze(x, 3)
        hidden_init = torch.zeros(self.num_layers, batch_size, self.lstm_out_dim).to(self.device)
        cell_init = torch.zeros(self.num_layers, batch_size, self.lstm_out_dim).to(self.device)

        output, hidden = self.lstm(x, (hidden_init, cell_init))
        output = output.unsqueeze(-1) # [b, L, n, 1]
    
        prediction = self.fc(output).permute(0,2,1,3) # [b, 1, n, 1] --> [b,n,1,1]
        prediction = self.fc2(prediction).transpose(1,3) 
                # [b,n,1,1] --> [b,9,1,1] --> [b,n,1,1] --> [b,1,1,9]
        prediction = prediction.squeeze()       
        
        return prediction         
        

