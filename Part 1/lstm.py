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

        self.W_gx = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.W_gh = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_g = nn.Parameter(torch.Tensor(hidden_dim))

        self.W_ix = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.W_ih = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_i = nn.Parameter(torch.Tensor(hidden_dim))

        self.W_fx = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.W_fh = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_f = nn.Parameter(torch.Tensor(hidden_dim))

        self.W_ox = nn.Parameter(torch.Tensor(input_dim, hidden_dim))
        self.W_oh = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.b_o = nn.Parameter(torch.Tensor(hidden_dim))

        self.W_ph = nn.Parameter(torch.Tensor(hidden_dim, output_dim))
        self.b_p = nn.Parameter(torch.Tensor(output_dim))

        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_uniform_(self.W_gx, a=0.1)
        nn.init.kaiming_uniform_(self.W_gh, a=0.1)
        nn.init.constant_(self.b_g, 0)

        nn.init.kaiming_uniform_(self.W_ix, a=0.1)
        nn.init.kaiming_uniform_(self.W_ih, a=0.1)
        nn.init.constant_(self.b_i, 0)

        nn.init.kaiming_uniform_(self.W_fx, a=0.1)
        nn.init.kaiming_uniform_(self.W_fh, a=0.1)
        nn.init.constant_(self.b_f, 0)

        nn.init.kaiming_uniform_(self.W_ox, a=0.1)
        nn.init.kaiming_uniform_(self.W_oh, a=0.1)
        nn.init.constant_(self.b_o, 0)

        nn.init.kaiming_uniform_(self.W_ph, a=0.1)
        nn.init.constant_(self.b_p, 0)

    ############
    def forward(self, x):
        # Implementation here ...
        ############
        batch_size = x.size(0)

        h_t = torch.zeros(batch_size, self.hidden_dim, dtype=torch.float32, device=x.device)
        c_t = torch.zeros(batch_size, self.hidden_dim, dtype=torch.float32, device=x.device)

        for t in range(self.seq_length):
            x_t = x[:, t, :]

            g_t = torch.tanh(torch.mm(x_t, self.W_gx) + torch.mm(h_t, self.W_gh) + self.b_g)
            i_t = torch.sigmoid(torch.mm(x_t, self.W_ix) + torch.mm(h_t, self.W_ih) + self.b_i)
            f_t = torch.sigmoid(torch.mm(x_t, self.W_fx) + torch.mm(h_t, self.W_fh) + self.b_f)
            o_t = torch.sigmoid(torch.mm(x_t, self.W_ox) + torch.mm(h_t, self.W_oh) + self.b_o)

            c_t = g_t * i_t + c_t * f_t
            h_t = torch.tanh(c_t) * o_t

        p_t = torch.mm(h_t, self.W_ph) + self.b_p
        y_t = nn.functional.softmax(p_t, dim=1)

        return y_t

    ############
    # add more methods here if needed
