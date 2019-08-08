import torch
import torch.nn as nn
import torch.optim as optim
import torchcontrib

from model import LSTM

class Trainer(nn.Module):
    def __init__(self,input_size, hidden_size, batch_size, output_dim, num_layers, learning_rate, seq_len):
        super(Trainer, self).__init__()
        self.model = LSTM(input_size, 
                          hidden_size, 
                          batch_size,
                          output_dim, 
                          num_layers,
                         seq_len)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.optimizer = torchcontrib.optim.SWA(self.optimizer, swa_start=10, swa_freq=5, swa_lr=0.05)
        self.criterion = nn.MSELoss()
        
    def learn(self, x_train, y_train):
        self.model.zero_grad()
        y_pred = self.model(x_train, True)
        
        loss = self.criterion(y_pred.t(), y_train)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
    
    def evaluate(self, x_test, y_test):
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(x_test, True)
            
        loss = self.criterion(y_pred.t(), y_test)
        self.model.train()
        return loss, y_pred.t()
    
    def predict(self, x):
        
        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(x, False)
        
        return y_pred.t()
        