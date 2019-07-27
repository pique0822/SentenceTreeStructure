import torch
from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop

import numpy as np

def get_model_values(model):
    params = {}
    for name, param in model.named_parameters():
        params[name] = param

    model_values = {}

    encoder_weights = params['encoder.weight']
    decoder_bias = params['decoder.bias']

    for layer in range(len(model.rnns)):

        hidden_size = model.rnns[layer].module.hidden_size
        name = 'rnns.'+str(layer)+'.module.'

        w_ii_l0 = params[name+'weight_ih_l0'][0:hidden_size]
        w_if_l0 = params[name+'weight_ih_l0'][hidden_size:2*hidden_size]
        w_ig_l0 = params[name+'weight_ih_l0'][2*hidden_size:3*hidden_size]
        w_io_l0 = params[name+'weight_ih_l0'][3*hidden_size:4*hidden_size]

        b_ii_l0 = params[name+'bias_ih_l0'][0:hidden_size]
        b_if_l0 = params[name+'bias_ih_l0'][hidden_size:2*hidden_size]
        b_ig_l0 = params[name+'bias_ih_l0'][2*hidden_size:3*hidden_size]
        b_io_l0 = params[name+'bias_ih_l0'][3*hidden_size:4*hidden_size]

        input_vals = (w_ii_l0,b_ii_l0,w_if_l0,b_if_l0,w_ig_l0,b_ig_l0,w_io_l0,b_io_l0)
        # Recurrent
        w_hi_l0 = params[name+'weight_hh_l0_raw'][0:hidden_size]
        w_hf_l0 = params[name+'weight_hh_l0_raw'][hidden_size:2*hidden_size]
        w_hg_l0 = params[name+'weight_hh_l0_raw'][2*hidden_size:3*hidden_size]
        w_ho_l0 = params[name+'weight_hh_l0_raw'][3*hidden_size:4*hidden_size]

        b_hi_l0 = params[name+'bias_hh_l0'][0:hidden_size]
        b_hf_l0 = params[name+'bias_hh_l0'][hidden_size:2*hidden_size]
        b_hg_l0 = params[name+'bias_hh_l0'][2*hidden_size:3*hidden_size]
        b_ho_l0 = params[name+'bias_hh_l0'][3*hidden_size:4*hidden_size]

        hidden_vals = (w_hi_l0,b_hi_l0,w_hf_l0,b_hf_l0,w_hg_l0,b_hg_l0,w_ho_l0,b_ho_l0)

        model_values[layer] = (input_vals,hidden_vals)

    return model_values

def back_prop_relevance(model, model_values, forget_gates, input_gates, output_gates, c_tilde_gates, hidden_states, cell_states):

    number_of_words = len(forget_gates)

    dhdx = {}
    dcdx = {}
    for word_index in range(number_of_words):

        for layer in range(len(forget_gates)):
            (input_vals,hidden_vals) = model_values[layer]

            (w_ii_l0,b_ii_l0,w_if_l0,b_if_l0,w_ig_l0,b_ig_l0,w_io_l0,b_io_l0) = input_vals
            (w_hi_l0,b_hi_l0,w_hf_l0,b_hf_l0,w_hg_l0,b_hg_l0,w_ho_l0,b_ho_l0) = hidden_vals

            f_gate = forget_gates[word_index][layer]
            i_gate = input_gates[word_index][layer]
            o_gate = output_gates[word_index][layer]
            c_tilde = c_tilde_gates[word_index][layer]

            h_state = hidden_states[word_index][layer]
            c_state = cell_states[word_index][layer]

            f_gate = torch.reshape(f_gate,(1,-1))
            i_gate = torch.reshape(i_gate,(1,-1))
            o_gate = torch.reshape(o_gate,(1,-1))
            c_tilde = torch.reshape(c_tilde,(1,-1))
            h_state = torch.reshape(h_state,(1,-1))
            c_state = torch.reshape(c_state,(1,-1))

            if layer == 0:


                df = torch.matmul(f_gate),(torch.ones(f_gate.shape) - f_gate)),w_if_l0)
                di = torch.matmul(torch.matmul(torch.t(i_gate),(torch.ones(i_gate.shape) - i_gate)),w_ii_l0)
                do = torch.matmul(torch.matmul(torch.t(o_gate),(torch.ones(o_gate.shape) - o_gate)),w_io_l0)

                dc_tilde = torch.matmul((1 - torch.matmul(torch.t(c_tilde),c_tilde)),w_ig_l0)
                import pdb; pdb.set_trace()

                torch.matmul(torch.t(c_state),(torch.ones(c_state.shape) - c_state))
                di * c_tilde
                i_gate * dc_tilde

                dc = torch.matmul(torch.matmul(torch.t(c_state),(torch.ones(c_state.shape) - c_state)), di * c_tilde + i_gate * dc_tilde)
                dh = ( torch.matmul((1 - torch.tanh(c_state)**2 ), dc), * o_gate ) + torch.tanh(c_state)*do

                dcdx[(word_index,layer)] = dc
                dhdx[(word_index,layer)] = dh
                import pdb; pdb.set_trace()

            else:
                df = torch.matmul(torch.matmul(f_gate,(torch.ones(f_gate.shape) - f_gate)),w_if_l0 + torch.matmul(dhdx[(word_index,layer-1)],w_hf_l0))
                di = torch.matmul(torch.matmul(i_gate,(torch.ones(i_gate.shape) - i_gate)),w_ii_l0 + torch.matmul(dhdx[(word_index,layer-1)],w_hi_l0))
                do = torch.matmul(torch.matmul(o_gate,(torch.ones(o_gate.shape) - o_gate)),w_io_l0 + torch.matmul(dhdx[(word_index,layer-1)],w_ho_l0))

                dc_tilde = torch.matmul((1 - torch.matmul(c_tilde,c_tilde)),w_ig_l0 + torch.matmul(dhdx[(word_index,layer-1)],w_hg_l0 + torch.matmul(dhdx[(word_index,layer-1)],w_hg_l0)))

                dc = torch.matmul(torch.matmul(c_state,(torch.ones(c_state.shape) - c_state)), di * c_tilde + i_gate * dc_tilde)

                dh = ( torch.matmul((1 - torch.tanh(c_state)**2 ), dc), * o_gate ) + torch.tanh(c_state)*do

                dcdx[(word_index,layer)] = dc
                dhdx[(word_index,layer)] = dh



    return result, hidden, out_gates, raw_outputs, outputs, emb

def gated_forward(model, model_values, input, hidden):

    emb = embedded_dropout(model.encoder, input, dropout=model.dropoute if model.training else 0)

    emb = model.lockdrop(emb, model.dropouti)

    raw_output = emb
    new_hidden = []
    #raw_output, hidden = model.rnn(emb, hidden)
    raw_outputs = []
    outputs = []

    out_gates = []

    for l, rnn in enumerate(model.rnns):
        # print('LAYER ',l)
        (h0_l0, c0_l0) = hidden[l]
        i_vals, h_vals = model_values[l]

        (w_ii_l0,b_ii_l0,w_if_l0,b_if_l0,w_ig_l0,b_ig_l0,w_io_l0,b_io_l0) = i_vals

        (w_hi_l0,b_hi_l0,w_hf_l0,b_hf_l0,w_hg_l0,b_hg_l0,w_ho_l0,b_ho_l0) = h_vals

        gated_out = []

        f_gates, i_gates, o_gates, g_gates = [],[],[],[]
        for seq_i in range(len(raw_output)):
            inp = raw_output[seq_i]

            # forget gate
            f_g_l0 = torch.sigmoid((torch.matmul(inp,torch.t(w_if_l0)) + b_if_l0) + (torch.matmul(h0_l0,torch.t(w_hf_l0)) + b_hf_l0))


            # input gate
            i_g_l0 = torch.sigmoid((torch.matmul(inp,torch.t(w_ii_l0)) + b_ii_l0) + (torch.matmul(h0_l0,torch.t(w_hi_l0)) + b_hi_l0))

            # output gate
            o_g_l0 = torch.sigmoid((torch.matmul(inp,torch.t(w_io_l0)) + b_io_l0) + (torch.matmul(h0_l0, torch.t(w_ho_l0)) + b_ho_l0))


            # intermediate cell state
            c_tilde_l0 = torch.tanh((torch.matmul(inp,torch.t(w_ig_l0)) + b_ig_l0) + (torch.matmul(h0_l0, torch.t(w_hg_l0)) + b_hg_l0))

            # current cell state
            c0_l0 = f_g_l0 * c0_l0 + i_g_l0 * c_tilde_l0

            # hidden state
            h0_l0 = o_g_l0 * torch.tanh(c0_l0)

            new_h = (h0_l0,c0_l0)
            gated_out.append(h0_l0)

            f_gates.append(f_g_l0)
            i_gates.append(i_g_l0)
            o_gates.append(o_g_l0)
            g_gates.append(c0_l0)

        gates = (f_gates,i_gates,o_gates,g_gates)
        out_gates.append(gates)
        # pdb.set_trace()
        out = torch.stack(gated_out).reshape(len(gated_out),h0_l0.shape[1],h0_l0.shape[2])
        raw_output = out

        new_hidden.append(new_h)
        raw_outputs.append(raw_output)
        if l != model.nlayers - 1:
            #model.hdrop(raw_output)
            raw_output = model.lockdrop(raw_output, model.dropouth)
            outputs.append(raw_output)
    hidden = new_hidden

    output = model.lockdrop(raw_output, model.dropout)
    outputs.append(output)

    result = output.view(output.size(0)*output.size(1), output.size(2))

    return result, hidden, out_gates, raw_outputs, outputs

def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def batchify(data, bsz, args):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data

def batchify_simple(data, bsz, cuda):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if cuda:
        data = data.cuda()
    return data


def get_batch(source, i, args, seq_len=None, evaluation=False):
    seq_len = min(seq_len if seq_len else args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target
