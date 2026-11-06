import torch
import os
from utils import perp_eval, get_wikitext2, tokenize, fetch_model

def quantize_rtn(W, n_bits=4, group_size=128):
    W3 = torch.reshape(W, (W.size(0), W.size(1)//group_size, group_size))
    W_d = torch.amax(W3.abs(), -1, keepdim = True) / pow(2,(n_bits-1))
    W_hat = W_d * torch.clamp(torch.round(W3/W_d), -pow(2,(n_bits-1)), pow(2,(n_bits-1))-1)
    return W_hat.reshape(W.shape)

def calibrate_awq(model, input_ids, seq_ct = 16, token_ct = 512):
    X = {}

    def make_hook(name):
        def calibration_hook(module, inputs, outputs):
            X[name].append(inputs[0].detach().half().reshape(-1, module.weight.shape[1]))

        return calibration_hook

    hook_handles = []

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and "layers." in name:
           X[name] = []
           handle = module.register_forward_hook(make_hook(name))
           hook_handles.append(handle)
    
    for i in range(seq_ct):
        seq_input_ids = input_ids[:, (i)*token_ct:(i+1)*token_ct]
        with torch.no_grad():
            outputs = model(seq_input_ids)

    for i in hook_handles:
        i.remove()

    s_X = {}
    for name in X:
        X[name] = torch.cat(X[name])
        s_X[name] = X[name].abs().float().mean(dim=0)

    return s_X, X

def search_alpha(W, X, s_x, n_bits = 4, group_size = 128, n_grid = 20):
    Xf = X.float()
    Wf = W.float()
    Y = Xf @ Wf.T

    a_list = []

    candidates = [i / (n_grid-1) for i in range(n_grid)]

    b_loss, b_alpha, b_s, b_W_eff = None, None, None, None

    for alpha in candidates:
        s = s_x.pow(alpha).clamp(min=1e-4)
        s = s / (s.max() * s.min()).sqrt()
        W_eff = quantize_rtn(Wf*s, n_bits, group_size) / s
        loss = ((Xf@W_eff.float().T)-Y).pow(2).mean().item()

        a_list.append((alpha, loss))

        if b_loss is None or b_loss > loss:
            b_loss = loss
            b_alpha = alpha
            b_s = s
            b_W_eff = W_eff
        
    return b_loss, b_alpha, b_s, a_list, b_W_eff

def apply_awq(model, X, s_X):
    res_dict = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and "layers." in name:
            with torch.no_grad():    
                i_X = X[name]
                i_W = module.weight
                i_s_X = s_X[name]

                loss, alpha, _, _ , i_W_eff = search_alpha(i_W, i_X, i_s_X)

                res_dict[name] = (alpha, loss)
                
                module.weight.copy_(i_W_eff.half())
    return res_dict
                
def test_quantize(encodings, X, s_X, max_segments=20):
    results = {}

    model = fetch_model()
    results["fp16"] = perp_eval(model, encodings["input_ids"], max_segments=max_segments)

    # RTN arm: reuse the same model, weights untouched by eval
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and "layers." in name:
            module.weight.data = quantize_rtn(module.weight.data)
    results["rtn"] = perp_eval(model, encodings["input_ids"], max_segments=max_segments)
    del model

    # AWQ arm: fresh weights, RTN arm already corrupted the old object
    model = fetch_model()
    alpha_dict = apply_awq(model, X, s_X)
    results["awq"] = perp_eval(model, encodings["input_ids"], max_segments=max_segments)

    return results, alpha_dict

def main():
    ds = get_wikitext2()
    encodings = tokenize(ds)
    model = fetch_model()

    if os.path.isfile("calib_cache.pt"):
        calib = torch.load("calib_cache.pt")
        s_X = calib["s_X"]
        X = calib["X"]
    else:
        s_X, X = calibrate_awq(model, encodings["input_ids"])
        torch.save({"X": X, "s_X": s_X}, "calib_cache.pt")

    results, alpha_dict = test_quantize(encodings, X, s_X)
    print(results)
    print(sorted(a for a, l in alpha_dict.values())[:10])
    print(max(alpha_dict.values()))



if __name__ == "__main__":
    main()