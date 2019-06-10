import torch
from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop

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
