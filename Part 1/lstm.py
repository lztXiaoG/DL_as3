from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


################################################################################

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, output_dim):
        super(LSTM, self).__init__()
        # Initialization here ...
        ############
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # LSTM parameters
        self.W_ih = nn.Parameter(torch.Tensor(input_dim, hidden_dim * 4))
        self.W_hh = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim * 4))
        self.b_ih = nn.Parameter(torch.Tensor(hidden_dim * 4))
        self.b_hh = nn.Parameter(torch.Tensor(hidden_dim * 4))

        # Output layer parameters
        self.W_out = nn.Parameter(torch.Tensor(hidden_dim, output_dim))
        self.b_out = nn.Parameter(torch.Tensor(output_dim))

    #     self.reset_parameters()
    #
    #     ############
    #
    # def reset_parameters(self):
    #     nn.init.kaiming_uniform_(self.W_ih, a=0.1)
    #     nn.init.kaiming_uniform_(self.W_hh, a=0.1)
    #     nn.init.constant_(self.b_ih, 0)
    #     nn.init.constant_(self.b_hh, 0)
    #     nn.init.kaiming_uniform_(self.W_out, a=0.1)
    #     nn.init.constant_(self.b_out, 0)

    ############
    def forward(self, x):
        # Implementation here ...
        ############
        batch_size = x.size(0)

        # Initialize hidden state and cell state
        h_t = torch.zeros(batch_size, self.hidden_dim, dtype=x.dtype, device=x.device)
        c_t = torch.zeros(batch_size, self.hidden_dim, dtype=x.dtype, device=x.device)

        # Iterate over sequence steps
        for t in range(self.seq_length):
            x_t = x[:, t, :]

            # LSTM recurrence equations
            gates = torch.mm(x_t, self.W_ih) + torch.mm(h_t, self.W_hh) + self.b_ih + self.b_hh
            i_t, f_t, g_t, o_t = gates.chunk(4, 1)
            i_t = torch.sigmoid(i_t)
            f_t = torch.sigmoid(f_t)
            g_t = torch.tanh(g_t)
            o_t = torch.sigmoid(o_t)

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

        # Compute output
        output = torch.mm(h_t, self.W_out) + self.b_out

        return output

    ############
    # add more methods here if needed
