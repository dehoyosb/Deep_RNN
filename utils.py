import torch
import torch.nn as nn
import numpy as np

def get_prediction_batch(x,y,shift,input_size,output_size, lookahead, trainer):
    predictions = []
    for i in range(round((lookahead-output_size+shift)/shift)):
        out_pred = trainer.predict(x)
        x_shifted = torch.roll(x, -shift)
        x_shifted[:,-shift:,:] = x[:,-shift:,:]
        for i in range(shift,0,-1):
            new_col = torch.roll(x[:,-i,:], -shift)
            if i == shift:
                new_col[-shift:] = out_pred[:shift,-1].unsqueeze(1)
                x_shifted[:,-i,:] = new_col
            else:
                new_col[-shift:] = torch.cat((x_shifted[-1:,-i+1],out_pred[:shift-1,-i+1].unsqueeze(1)))
                x_shifted[:,-i,:] = new_col
        x = x_shifted
        predictions.append(out_pred[:,-1])
    return x, predictions
    
def get_prediction(x, shift ,output_size, lookahead, trainer):
    outputs = []
    init_hidden = 479
    for i in range(round(lookahead/output_size) + init_hidden):
        if i == 0:
            trainer.model.hidden = (torch.zeros(trainer.model.num_layers,
                                                1,
                                                trainer.model.hidden_dim),
                                    torch.zeros(trainer.model.num_layers,
                                                1,
                                                trainer.model.hidden_dim))
        if i < init_hidden + 1:
            input_model = x[:,i-(init_hidden+1)]
            output = trainer.predict(input_model.unsqueeze(2))
        else:
            input_model = torch.roll(input_model,-output_size)
            input_model[-output.shape[0]:] = output
            output = trainer.predict(input_model.unsqueeze(2))
        outputs += [output]
        
    return torch.cat(outputs[init_hidden:]).numpy()
    
def extend(x, lookahead):
    diff_x = x[-1]-x[-2]
    x_new = x[-1] + diff_x
    extended = []

    for i in range(lookahead-1):
        extended.append(x_new)
        x_new += diff_x

    extended = np.array(extended)
    
    return np.append(x, extended)