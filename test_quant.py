import torch
from quant import quantize_rtn
from quant import pack_int4, unpack_int4

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

def test_pack_roundtrip():
    codes = torch.randint(-8, 8, (4, 2, 128), dtype=torch.int32)
    assert torch.equal(unpack_int4(pack_int4(codes)), codes)

def test_parts_match_dequant():
    W = torch.randn(4, 256)
    codes, W_d = quantize_rtn(W, 4, 128, return_parts=True)
    codes_rt = unpack_int4(pack_int4(codes))
    assert torch.equal((W_d * codes_rt.float()).reshape(W.shape), quantize_rtn(W, 4, 128))