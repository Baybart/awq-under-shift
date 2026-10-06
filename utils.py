import torch

import math
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model being evaluated
MODEL = "Qwen/Qwen2.5-0.5B"

def get_wikitext2():
    test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split = "test")
    print(test.column_names)
    # Joining in the same manner as AWQ paper to get matching results
    # We join rows into a single line, seperated by "\n\n"
    joined_test = "\n\n".join(test["text"])

    return joined_test

def tokenize(dataset):
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    encodings = tokenizer(dataset, return_tensors = "pt")
    
    return encodings

def fetch_model():
    if MODEL == "Qwen/Qwen2.5-0.5B":
        return AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")

    
# Note this this eval is done with stride = seqlen
# While this is suboptimal, we're matching AWQ's eval configuration
def perp_eval(model, input_ids, seqlen = 2048, max_segments = None):
    # We throw away the last incomplete sequence
    token_ct = (len(input_ids[0])//seqlen)*seqlen

    seq_id = 0

    nll_sum = 0.0
    while(seq_id*seqlen < token_ct and (max_segments is None or seq_id < max_segments)):
        print(seq_id)
        seq_input_ids = input_ids[:, seq_id*seqlen:(seq_id+1)*seqlen]

        with torch.no_grad():
            outputs = model(seq_input_ids, labels=seq_input_ids)
            neg_log_likelihood = outputs.loss.item()

        nll_sum += neg_log_likelihood * seqlen

        seq_id += 1

    n_tokens = token_ct if max_segments is None else seqlen*seq_id

    avg_nll = nll_sum / n_tokens
    ppl = math.exp(avg_nll)

    return ppl




def main():
    ds = get_wikitext2()
    encodings = tokenize(ds)
    model = fetch_model()

    perp = perp_eval(model = model, input_ids = encodings["input_ids"], max_segments = 1)
    print(f"perplexity: {perp}")

if __name__ == "__main__":
    main()
