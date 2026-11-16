import json, torch
from quant import calibrate_awq, apply_awq, quantize_rtn
from utils import fetch_model, get_wikitext2, get_domain, tokenize, perp_eval

DOMAINS = ["pubmed", "enron", "freelaw", "github"]
SEGS = 100  # eval segments per number
DEV = "cuda"

MODEL = "meta-llama/Llama-3.2-3B"

def save(results, path="results.json"):
    with open(path, "w") as f:
        json.dump(results, f, indent=2)

def linear_modules(model):
    return [(n, m) for n, m in model.named_modules()
            if isinstance(m, torch.nn.Linear) and "layers." in n]

def run():
    results = {}

    # tokenize every eval set once
    evals = {"wikitext": tokenize(get_wikitext2("test"), MODEL)["input_ids"].to(DEV)}
    for d in DOMAINS:
        evals[d] = tokenize(get_domain(d, "eval"), MODEL)["input_ids"].to(DEV)

    # FP16 row
    model = fetch_model(MODEL)
    for name, ids in evals.items():
        results[f"fp16/{name}"] = perp_eval(model, ids, max_segments=SEGS)
        print(f"fp16/{name}: {results[f'fp16/{name}']:.3f}"); save(results)

    calib_wiki = tokenize(get_wikitext2("train"), MODEL)["input_ids"].to(DEV)
    s_X, X = calibrate_awq(model, calib_wiki)

    # RTN row
    for n, m in linear_modules(model):
        m.weight.data = quantize_rtn(m.weight.data)
    for name, ids in evals.items():
        results[f"rtn/{name}"] = perp_eval(model, ids, max_segments=SEGS)
        print(f"rtn/{name}: {results[f'rtn/{name}']:.3f}"); save(results)
    del model; torch.cuda.empty_cache()

    # AWQ generic-calibration row
    model = fetch_model(MODEL)
    apply_awq(model, X, s_X)
    del X, s_X
    for name, ids in evals.items():
        results[f"awq_wiki/{name}"] = perp_eval(model, ids, max_segments=SEGS)
        print(f"awq_wiki/{name}: {results[f'awq_wiki/{name}']:.3f}"); save(results)
    del model; torch.cuda.empty_cache()

    # AWQ matched-calibration diagonal
    for d in DOMAINS:
        model = fetch_model(MODEL)
        calib_ids = tokenize(get_domain(d, "calib"), MODEL)["input_ids"].to(DEV)
        s_X, X = calibrate_awq(model, calib_ids)
        apply_awq(model, X, s_X)
        del X, s_X
        results[f"awq_{d}/{d}"] = perp_eval(model, evals[d], max_segments=SEGS)
        print(f"awq_{d}/{d}: {results[f'awq_{d}/{d}']:.3f}"); save(results)
        del model; torch.cuda.empty_cache()

if __name__ == "__main__":
    run()