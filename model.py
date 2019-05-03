import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim, num_layers, seq_len):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.seq_len = seq_len

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, dropout = 0.2)
        
        self.drop_out = nn.Dropout(0.2)

        # Define the output layer
        self.linear1 = nn.Linear(self.hidden_dim*self.seq_len, self.output_dim[0])
        self.linear2 = nn.Linear(self.output_dim[0], self.output_dim[1])
        self.linear3 = nn.Linear(self.output_dim[1], self.output_dim[2])

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers,self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers,self.batch_size, self.hidden_dim))

    def forward(self, x, batch):
        
        if batch:
            outputs = []
            self.hidden = self.init_hidden()

            for i, lstm_input in enumerate(x.chunk(round(x.size(1)/self.batch_size), dim=1)):
                # Forward pass through LSTM layer
                # shape of lstm_out: [input_size, batch_size, hidden_dim]
                # shape of self.hidden: (h, c), where h and c both 
                # have shape (num_layers, batch_size, hidden_dim).

                output, self.hidden = self.lstm(lstm_input, self.hidden)

                # Output from lstm is [seq_len, batch_size, hidden_dim], but for the linear's input we need
                # [batch_size, hidden_dim*seq_len]. For this, we first swap the batch dimention with permute,
                # and then we apply a flatten with the last 2 dimentions with view.

                input_linear = output.permute(1, 0, 2).contiguous()
                input_linear = input_linear.view(self.batch_size,-1)    
                y_pred = self.linear1(input_linear)
                y_pred = self.linear2(y_pred)
                y_pred = self.linear3(y_pred)
                outputs += [y_pred]

            # At the end, we just concatenate the predictions from the batch loop, and configure them to be 
            # [full_input_dim, output_dim], using view.

            outputs = torch.stack(outputs).view(-1,self.output_dim[2])
        else:
            output, self.hidden = self.lstm(x, self.hidden)
            input_linear = output.permute(1, 0, 2).contiguous()
            input_linear = input_linear.view(1,-1)    
            y_pred = self.linear1(input_linear)
            y_pred = self.linear2(y_pred)
            outputs = self.linear3(y_pred)
            
        return outputs