import os

import transformers
import datasets
import torch
import json
from typing import Iterable, List
from tqdm.auto import tqdm

def read_prompts_from_file(path: str) -> List[str]:
    prompts = []
    if path.endswith(".jsonl"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    # support either {'text': ...} or {'prompt': ...}
                    text = obj.get("text") or obj.get("prompt") or obj.get("input")
                    if text is None:
                        # fallback to whole object repr
                        text = json.dumps(obj, ensure_ascii=False)
                except Exception:
                    text = line
                prompts.append(text)
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    prompts.append(line)

    return prompts

def normalize_input_length(prompts: List[str], tokenizer, target_length: int = 128) -> List[str]:
    """Filter prompts with at least target_length tokens and truncate to exactly target_length."""
    normalized_prompts = []
    
    print(f"\nFiltering prompts with at least {target_length} tokens and truncating to exactly {target_length}...")
    
    filtered_count = 0
    for prompt in tqdm(prompts, desc="Processing inputs"):
        # Tokenize the prompt
        tokens = tokenizer.encode(prompt, add_special_tokens=False)
        
        # Only keep prompts with at least target_length tokens
        if len(tokens) >= target_length:
            # Truncate to exactly target_length tokens
            tokens = tokens[:target_length]
            
            # Decode back to text
            normalized_text = tokenizer.decode(tokens, skip_special_tokens=False)
            normalized_prompts.append(normalized_text)
        else:
            filtered_count += 1
    
    print(f"✅ Kept {len(normalized_prompts)} prompts with exactly {target_length} tokens")
    print(f"   Filtered out {filtered_count} prompts with fewer than {target_length} tokens")
    return normalized_prompts


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    prompts = read_prompts_from_file("data/imdb_test.jsonl")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    prompts = normalize_input_length(prompts, tokenizer, target_length=128)

    ## Model Loading with flashattention enabled and bf16
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation='eager',
    )
    try:
        model.to(device)
    except Exception as e:
        print(f"Error moving model to device: {e}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"  Warning: pad_token was None, set to eos_token ({tokenizer.eos_token})")

    # batch_list = [4]
    cuda_graphs = {}

    # batch_list = [1,2,4,8,16,32]
    batch_list = [b for b in range(1, 17, 1)]
    for batch_size in batch_list:
        print("Process Capture CUDA Graph for batch size:", batch_size)
        past_kv_values = None
        original_padding_side = tokenizer.padding_side
        tokenizer.padding_side = 'left'
        inputs = tokenizer(prompts[:batch_size], return_tensors="pt", padding=True).to(device)
        tokenizer.padding_side = original_padding_side
        prefill_test = []
        # print(inputs)
        
        with torch.no_grad():
            prefill_outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=True,
                return_dict=True,
            )
        torch.cuda.synchronize()

        # Pick next token from prefill logits
        next_token_logits = prefill_outputs.logits[:, -1, :]
        next_token_ids = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        for i, text in enumerate(next_token_ids):
            decoded_texts = tokenizer.batch_decode(text, skip_special_tokens=True)
            # print(f"input: {prompts[i]}")
            # print(f"Decoded text for batch {i}: {decoded_texts}")
            prefill_test.append(decoded_texts)

        past_key_values = prefill_outputs.past_key_values

        # print(f"Prefill attention mask shape: {inputs['attention_mask'].shape}")
        # print("=" * 60)

        # 2) 첫 Decoding Warmup
        attn_mask_dec = torch.cat(
            [inputs["attention_mask"], torch.ones((inputs["attention_mask"].size(0), 1), device=inputs["attention_mask"].device, dtype=inputs["attention_mask"].dtype)],
            dim=-1,
        )

        # change attn_mask_dec to True/False
        attn_mask_dec = attn_mask_dec.bool()
        # print(f"Decoding attention mask shape: {attn_mask_dec.shape}")
        # print("=" * 60)
        min_dtype = torch.finfo(torch.bfloat16).min
        mask = torch.where(attn_mask_dec, torch.tensor(0.0, device=attn_mask_dec.device, dtype=torch.bfloat16), min_dtype)
        outputs_idx = None
        for _ in range(2):
            with torch.no_grad():
                outputs_idx = model(
                    input_ids=next_token_ids,
                    attention_mask=mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
        torch.cuda.synchronize()
        # print decoding tokens text
        next_token_logits = outputs_idx.logits[:, -1, :]
        outputs_idx = torch.argmax(next_token_logits, dim=-1, keepdim=False)

        # print(f"Decoding output token ids: {outputs_idx}")

        # print decoding tokens text
        
        for i, text in enumerate(outputs_idx):
            decoded_texts = tokenizer.batch_decode(text, skip_special_tokens=True)
            # print(f"input: {prompts[i]}{prefill_test[i][0]}")
            # print(f"Decoded text for batch {i}: {decoded_texts}")
            prefill_test.append(decoded_texts)
        # print("=" * 60)
        
        # ones = torch.ones_like(next_token_ids).to(device)
        # next_token_ids = torch.add(next_token_ids, ones)
        # print(f"Next token ids for CUDA graph capture: {next_token_ids}")

        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph):
            with torch.no_grad():
                outputs_idx = model(
                    input_ids=next_token_ids,
                    attention_mask=mask,
                    past_key_values = past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
        

        graph.replay()
        # print(f"Replayed CUDA graph for batch size {batch_size}")

        ## print replay output tokens
        next_token_logits = outputs_idx.logits[:, -1, :]
        outputs_idx = torch.argmax(next_token_logits, dim=-1, keepdim=False)
        # print(f"Decoding output token ids after graph replay: {outputs_idx}")
        
        for i, text in enumerate(outputs_idx):
            decoded_texts = tokenizer.batch_decode(text, skip_special_tokens=True)
            # print(f"input: {prompts[i]}{prefill_test[i][0]}")
            # print(f"Decoded text for batch {i}: {decoded_texts[0]}")
        info_dict = {
            'input_text': "{prompts[i]}{prefill_test[i][0]}",
            'mask': mask,
            'graph': graph,
        }
        cuda_graphs[batch_size] = info_dict
        graph.enable_debug_mode()

        os.makedirs(f"cuda_graphs", exist_ok=True)
        file_path = f"cuda_graphs/base_llama3_8b_bs_{batch_size}_graph.pt"
        graph.debug_dump(file_path)

    # for batch_size, graph in cuda_graphs.items():
    #     print(f"Captured CUDA graph for batch size {batch_size}")
        # print(f"Captured CUDA graph for batch size {batch_size}")


if __name__ == "__main__":
    main()