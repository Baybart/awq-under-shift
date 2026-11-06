import torch
from quant import quantize_rtn

def group_quant(w, n_bits):
    delta = w.abs().max() / pow(2,(n_bits-1))
    # fake quantization, we dequantize immediatly
    w_hat = delta * torch.clamp(torch.round(w/delta), -pow(2,(n_bits-1)), pow(2,(n_bits-1))-1)
    return w_hat

def test_rtn_matches_scalar():
    W = torch.randn(4, 256)
    W_hat = quantize_rtn(W, 4, 128)
    for i in range(4):
        for j in range(2):
            gq_out = group_quant(W[i][j*128:(j+1)*128], 4)
            rtn_out = W_hat[i][j*128:(j+1)*128]
            assert torch.equal(gq_out, rtn_out)