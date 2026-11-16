import torch

import math
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


DOMAINS = {
    "pubmed":  ("PubMed Abstracts", "text"),
    "enron":   ("Enron Emails", "text"),
    "freelaw": ("FreeLaw", "text"),
    "github":  ("Github", "text"),
}

def get_domain(name, kind):
    subset, fields = DOMAINS[name]

    ds = load_dataset("ArmelR/the-pile-splitted", subset, split="train", streaming=True)

    budget = 4000000

    texts, total = [], 0

    if kind == "calib":
        for row in ds.take(200):
            texts.append(row["text"])
    else:
        for row in ds.skip(200):
            texts.append(row["text"])
            total += (len(row["text"]))
            if total >= budget:
                break
    
    joined_text = "\n\n".join(texts)
    return joined_text



def get_wikitext2(split = "test"):
    text = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split = split)
    # print(test.column_names)
    # Joining in the same manner as AWQ paper to get matching results
    # We join rows into a single line, seperated by "\n\n"
    joined_text = "\n\n".join(text["text"])

    return joined_text

def tokenize(dataset, name):
    tokenizer = AutoTokenizer.from_pretrained(name)
    encodings = tokenizer(dataset, return_tensors = "pt")
    
    return encodings

def fetch_model(name="Qwen/Qwen2.5-0.5B"):
    return AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16, device_map={"": 0})

    
# Note this this eval is done with stride = seqlen
# While this is suboptimal, we're matching AWQ's eval configuration
def perp_eval(model, input_ids, seqlen = 2048, max_segments = None):
    # We throw away the last incomplete sequence
    token_ct = (len(input_ids[0])//seqlen)*seqlen

    seq_id = 0

    nll_sum = 0.0
    while(seq_id*seqlen < token_ct and (max_segments is None or seq_id < max_segments)):
        # print(seq_id)
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
