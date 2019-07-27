import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from enum import IntEnum

class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

class GatedLSTM(nn.Module):
    def __init__(self,input_size,hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # input gate
        self.W_ii = Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hi = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = Parameter(torch.Tensor(hidden_size))
        # forget gate
        self.W_if = Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hf = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = Parameter(torch.Tensor(hidden_size))
        # ???
        self.W_ig = Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hg = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_g = Parameter(torch.Tensor(hidden_size))
        # output gate
        self.W_io = Parameter(torch.Tensor(input_size, hidden_size))
        self.W_ho = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = Parameter(torch.Tensor(hidden_size))

        self.decoder = nn.Linear(hidden_size, input_size)

        self.init_weights()
    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, x, init_states):
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        hidden_seq = []

        forget_gates = []
        input_gates = []
        output_gates = []
        intermediate_states = []
        cell_states = []

        # if not initial states are passed in, we will generate a hidden_state
        if init_states is None:
            h_t, c_t = torch.zeros(self.hidden_size).to(x.device), torch.zeros(self.hidden_size).to(x.device)
        else:
            h_t, c_t = init_states

        for t in range(seq_sz): # iterate over the time steps
            x_t = x[:, t, :]

            i_t = torch.sigmoid(x_t @ self.W_ii + h_t @ self.W_hi + self.b_i)

            f_t = torch.sigmoid(x_t @ self.W_if + h_t @ self.W_hf + self.b_f)

            g_t = torch.tanh(x_t @ self.W_ig + h_t @ self.W_hg + self.b_g)

            o_t = torch.sigmoid(x_t @ self.W_io + h_t @ self.W_ho + self.b_o)

            c_t = f_t * c_t + i_t * g_t

            h_t = o_t * torch.tanh(c_t)

            hidden_seq.append(h_t.unsqueeze(Dim.batch))

            forget_gates.append(f_t.unsqueeze(Dim.batch))
            input_gates.append(i_t.unsqueeze(Dim.batch))
            output_gates.append(o_t.unsqueeze(Dim.batch))
            intermediate_states.append(g_t.unsqueeze(Dim.batch))
            cell_states.append(c_t.unsqueeze(Dim.batch))

        hidden_seq = torch.cat(hidden_seq, dim=Dim.batch)
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()

        return hidden_seq, (h_t, c_t), (forget_gates, input_gates, output_gates, intermediate_states, cell_states)
