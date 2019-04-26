import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, output_dim, num_layers):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Define the LSTM layer
#         self.lstm = nn.LSTMCell(self.input_dim, self.hidden_dim, self.num_layers, batch_first = True)
        self.lstm_1 = nn.LSTMCell(self.input_dim, self.hidden_dim)
        self.lstm_2 = nn.LSTMCell(self.hidden_dim, self.hidden_dim)
#         self.hidden = self.init_hidden()

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)

#     def init_hidden(self):
#         # This is what we'll initialise our hidden state as
#         return (torch.zeros(self.batch_size, self.hidden_dim),
#                 torch.zeros(self.batch_size, self.hidden_dim))

    def forward(self, x):
        outputs = []
        for i, lstm_input in enumerate(x.chunk(round(x.size(0)/self.batch_size), dim=0)):
            # Forward pass through LSTM layer
            # shape of lstm_out: [input_size, batch_size, hidden_dim]
            # shape of self.hidden: (h, c), where a and b both 
            # have shape (num_layers, batch_size, hidden_dim).
            if i == 0:
                output_1, self.hidden = self.lstm_1(lstm_input)
                output_2, self.hidden = self.lstm_1(output_1)
                
            output_1, self.hidden = self.lstm_1(lstm_input, self.hidden)
            output_2, self.hidden = self.lstm_1(output_1, self.hidden)
                
            print(output.shape)
            
            # Only take the output from the final timetep
            # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
#             y_pred = self.linear(output[-1].view(self.batch_size, -1))
            y_pred = self.linear(output_2)
            outputs += [y_pred]
#         outputs = torch.stack(outputs).view(x.shape[0],-1,self.output_dim)
        return outputs