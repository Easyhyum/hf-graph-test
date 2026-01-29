import transformers
import datasets
import torch
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

    batch_list = [4]
    # batch_list = [1,2,4,8,16,32]
    # batch_list = [b for b in range(1, 17, 1)]
    for batch_size in batch_list:
        # warmup and graph capture
        for _ in range(2):
            original_padding_side = tokenizer.padding_side
            tokenizer.padding_side = 'left'
            inputs = tokenizer(prompts[:batch_size], return_tensors="pt", padding=True, pad_token_id=pad_token_id).to(device)
            tokenizer.padding_side = original_padding_side
            
            print(inputs)

            # with torch.no_grad():
            #     outputs = model(
            #         input_ids=inputs["input_ids"],
            #         attention_mask=inputs["attention_mask"],
            #         use_cache=True,
            #         return_dict=True,
            #     )


if __name__ == "__main__":
    main()