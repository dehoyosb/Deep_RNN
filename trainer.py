import torch
import torch.nn as nn
import torch.optim as optim

from model import LSTM

class Trainer(nn.Module):
    def __init__(self,input_size, hidden_size, batch_size, output_dim, num_layers, learning_rate):
        super(Trainer, self).__init__()
        self.model = LSTM(input_size, 
                          hidden_size, 
                          batch_size,
                          output_dim, 
                          num_layers)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
    def predict(self, x_train):
        return self.model(x_train)
        
    def learn(self, y_pred, y_train):
        self.model.zero_grad()
#         self.model.hidden = self.model.init_hidden()
        loss = self.criterion(y_pred, y_train)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()