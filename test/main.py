import os, time

import transformers, argparse
import datasets
import torch
import json
from typing import Iterable, List
from tqdm.auto import tqdm

import pynvml

from gpu_profile import GPUMonitor

import cuda_graph_api

clock_changed = False

def set_specific_clock(handle, TARGET_MEM_CLOCK = 9001, TARGET_SM_CLOCK = 2460, device_index=0):
    global clock_changed
    # print(pynvml.__version__)
    # print(f" Pynvml Version {pynvml.nvmlSystemGetNVMLVersion().decode('utf-8')}")
    try:
        device_name = pynvml.nvmlDeviceGetName(handle)
        
        print(f"=== Configuring GPU: {device_name} ===")
        print(f"Target Memory Clock: {TARGET_MEM_CLOCK} MHz")
        print(f"Target SM Clock    : {TARGET_SM_CLOCK} MHz")
        # applied_clocks = pynvml.nvmlDeviceGetApplicationsClock(handle, pynvml.NVML_CLOCK_SM)

        # print(f"--> Before Application Clock Setting: {applied_clocks} MHz")

        # 1. 클럭 설정 (Root 권한 필요)
        # pynvml.nvmlDeviceSetApplicationsClocks(handle, TARGET_MEM_CLOCK, TARGET_SM_CLOCK)
        
        # print("\n✅ Successfully set application clocks!")

        # 2. 적용 확인
        # 주의: 설정 직후에는 부하가 없으면 클럭이 낮게 보일 수 있습니다 (P-State 절전).
        # 확실한 확인을 위해 설정을 '조회'합니다.
        # applied_clocks = pynvml.nvmlDeviceGetApplicationsClock(handle, pynvml.NVML_CLOCK_SM)
        # print(f"--> After Application Clock Setting: {applied_clocks} MHz")

        pynvml.nvmlDeviceSetMemoryLockedClocks(handle, TARGET_MEM_CLOCK, TARGET_MEM_CLOCK)
        print("\n✅ Successfully set memory locked clocks!")

        pynvml.nvmlDeviceSetGpuLockedClocks(handle, TARGET_SM_CLOCK, TARGET_SM_CLOCK)
        print("\n✅ Successfully set application clocks!")

        # min_clock, max_clock = pynvml.nvmlDeviceGetGpuLockedClocks(handle)
        # print(f"🔒 Locked Setting (Range): {min_clock} MHz ~ {max_clock} MHz")

        # SM Clock Information
        try:
            sm_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
            print(f"SM Clock: {sm_clock} MHz")
        except Exception as e:
            print(f"SM Clock: Error - {e}")
        
        # Memory Clock Information
        try:
            mem_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            print(f"Memory Clock: {mem_clock} MHz")
        except Exception as e:
            print(f"Memory Clock: Error - {e}")
        
        clock_changed = True
    except pynvml.NVMLError as e:
        import traceback
        traceback.print_exc()
        raise SystemExit(f"Critical NVML Error: {e}")

global_sm_clock_supported = {}
target_test_clock = []

def print_supported_clocks(handle, device_index=0):
    global global_sm_clock_supported
    global target_test_clock
    TARGET_SM_CLOCKS_IN_MEMORY_CLOCKS = 5
    try:
        device_name = pynvml.nvmlDeviceGetName(handle)
        
        print(f"=== GPU: {device_name} (Index: {device_index}) ===")
        
        max_boost_clock = pynvml.nvmlDeviceGetMaxClockInfo(handle,  pynvml.NVML_CLOCK_SM)
        print(f"Max Boost Graphics Clock: {max_boost_clock} MHz\n")

        try:
            max_customer_sm_clock = pynvml.nvmlDeviceGetMaxCustomerBoostClock(handle, pynvml.NVML_CLOCK_SM)

            print(f"Max Customer SM Clock: {max_customer_sm_clock} MHz\n")
        except pynvml.NVMLError as e:
            print(f"Could not get Max Customer SM Clock: {e}\n")
        print(f"Fetching supported clock combinations...\n")
        print(f"Print All Temerature Thresholds:")

        try:
            power_state = pynvml.nvmlDeviceGetPowerState(handle)
            print(f" - Current Power-State: P{power_state}\n")
        except pynvml.NVMLError as e:
            print(f"   - Error fetching Power-State: {e}\n")

        try:
            performance_state = pynvml.nvmlDeviceGetPerformanceState(handle)
            print(f" - Current Performance-State: P{performance_state}\n")
        except pynvml.NVMLError as e:
            print(f"   - Error fetching Performance-State: {e}\n")

        try:
            power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
            print(f" - Current Power Limit: {power_limit:.2f} W\n")
        except pynvml.NVMLError as e:
            print(f"   - Error fetching Power Limit: {e}\n")
        
        try:
            power_limit_default = pynvml.nvmlDeviceGetPowerManagementDefaultLimit(handle) / 1000.0
            print(f" - Default Power Limit: {power_limit_default:.2f} W\n")        
        except pynvml.NVMLError as e:
            print(f"   - Error fetching Default Power Limit: {e}\n")    
        
        try:
            power_limit_constraints = pynvml.nvmlDeviceGetPowerManagementLimitConstraints(handle)
            print(f" - Power Limit Constraints: Min {power_limit_constraints[0]/1000.0:.2f} W, Max {power_limit_constraints[1]/1000.0:.2f} W\n")        
        except pynvml.NVMLError as e:
            print(f"   - Error fetching Power Limit Constraints: {e}\n")

        # 1. 온도 임계치 정보 출력

        try:
            try:
                shutdown_temp_threshold = pynvml.nvmlDeviceGetTemperatureThreshold(handle, pynvml.NVML_TEMPERATURE_THRESHOLD_SHUTDOWN)
                print(f"   - Shutdown Temperature Threshold: {shutdown_temp_threshold} °C")
            except pynvml.NVMLError as e:
                print(f"   - Error fetching shutdown temperature threshold: {e}")
            try:
                slow_temp_threshold = pynvml.nvmlDeviceGetTemperatureThreshold(handle, pynvml.NVML_TEMPERATURE_THRESHOLD_SLOWDOWN)
                print(f"   - Slowdown Temperature Threshold: {slow_temp_threshold} °C")
            except pynvml.NVMLError as e:
                print(f"   - Error fetching slowdown temperature threshold: {e}")
            try:
                memmax_temp_threshold = pynvml.nvmlDeviceGetTemperatureThreshold(handle, pynvml.NVML_TEMPERATURE_THRESHOLD_MEM_MAX)
                print(f"   - Mem Max Temperature Threshold: {memmax_temp_threshold} °C")
            except pynvml.NVMLError as e:
                print(f"   - Error fetching mem max temperature threshold: {e}")
            try:
                gpumax_temp_threshold = pynvml.nvmlDeviceGetTemperatureThreshold(handle, pynvml.NVML_TEMPERATURE_THRESHOLD_GPU_MAX)
                print(f"   - GPU Max Temperature Threshold: {gpumax_temp_threshold} °C")
            except pynvml.NVMLError as e:
                print(f"   - Error fetching gpu max temperature threshold: {e}")

        except pynvml.NVMLError as e:
            print(f"   - Error fetching temperature threshold: {e}")

        # 2. 지원되는 모든 Memory Clock 목록 가져오기
        try:
            mem_clocks = pynvml.nvmlDeviceGetSupportedMemoryClocks(handle)
        except pynvml.NVMLError as e:
            print(f"Error getting memory clocks: {e}")
            return

        if not mem_clocks:
            print("No supported memory clocks found.")
            return
        else:
            print(f"Total Supported Memory Clocks: {len(mem_clocks)}")
            print(f"Memory Clocks: {mem_clocks}\n")
        # 3. 각 Memory Clock에 대해 반복
        for m_clk in mem_clocks:
            global_sm_clock_supported[m_clk] = []
            print(f"▶ Memory Clock: {m_clk} MHz")
            
            try:
                # 해당 Memory Clock일 때 지원되는 SM(Graphics) Clock 목록 조회
                sm_clocks = pynvml.nvmlDeviceGetSupportedGraphicsClocks(handle, m_clk)
                
                # 결과 출력
                if sm_clocks:
                    # 너무 길어질 수 있으므로 개수와, 최대/최소, 그리고 전체 리스트 출력
                    print(f"   - Count: {len(sm_clocks)} supported SM clocks")
                    print(f"   - Range: {min(sm_clocks)} MHz ~ {max(sm_clocks)} MHz")
                    print(f"   - All SM Clocks: {sm_clocks}")
                    global_sm_clock_supported[m_clk] = sm_clocks
                else:
                    print("   - No supported SM clocks for this memory frequency.")
                    global_sm_clock_supported[m_clk] = []
            except pynvml.NVMLError as e:
                # 특정 메모리 클럭 조합에서 에러가 날 경우 예외 처리
                print(f"   - Error fetching SM clocks: {e}")
            
            print("-" * 60) # 구분선

    except pynvml.NVMLError as e:
        print(f"Critical NVML Error: {e}")
        
        

def print_gpu_info(handle):
    """Print comprehensive GPU information including clocks"""
    try:
        # Get GPU name
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        print(f"\n{'='*60}")
        print(f"GPU: {gpu_name}")
        print(f"{'='*60}")
        
        # SM Clock Information
        try:
            sm_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
            print(f"SM Clock: {sm_clock} MHz")
        except Exception as e:
            print(f"SM Clock: Error - {e}")
        
        # Memory Clock Information
        try:
            mem_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            print(f"Memory Clock: {mem_clock} MHz")
        except Exception as e:
            print(f"Memory Clock: Error - {e}")
        
        # Graphics Clock Information
        try:
            graphics_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)
            print(f"Graphics Clock: {graphics_clock} MHz")
        except Exception as e:
            print(f"Graphics Clock: Error - {e}")
        
        # Max Clock Information
        try:
            max_sm_clock = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM)
            max_mem_clock = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            print(f"Max SM Clock: {max_sm_clock} MHz")
            print(f"Max Memory Clock: {max_mem_clock} MHz")
        except Exception as e:
            print(f"Max Clocks: Error - {e}")
        
        # Temperature
        try:
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            print(f"Temperature: {temp}°C")
        except Exception as e:
            print(f"Temperature: Error - {e}")
        
        # Power Usage
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert mW to W
            power_limit = pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
            print(f"Power Usage: {power:.2f} W / {power_limit:.2f} W")
        except Exception as e:
            print(f"Power: Error - {e}")
        
        # Utilization
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            print(f"GPU Utilization: {util.gpu}%")
            print(f"Memory Utilization: {util.memory}%")
        except Exception as e:
            print(f"Utilization: Error - {e}")
        
        # Memory Info
        try:
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used_gb = mem_info.used / (1024**3)
            total_gb = mem_info.total / (1024**3)
            print(f"Memory: {used_gb:.2f} GB / {total_gb:.2f} GB ({mem_info.used * 100 / mem_info.total:.1f}%)")
        except Exception as e:
            print(f"Memory Info: Error - {e}")
        
        # PCIe Throughput
        try:
            pcie_tx = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_TX_BYTES)
            pcie_rx = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_RX_BYTES)
            print(f"PCIe TX: {pcie_tx} KB/s")
            print(f"PCIe RX: {pcie_rx} KB/s")
        except Exception as e:
            print(f"PCIe Throughput: Error - {e}")
        
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"Error getting GPU info: {e}")

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


def copy_cache_tensors(dst_cache, src_cache, batch_size: int | None = None) -> None:
    if dst_cache is None or src_cache is None:
        return
    if not hasattr(dst_cache, "layers") or not hasattr(src_cache, "layers"):
        return

    for src_layer, dst_layer in zip(src_cache.layers, dst_cache.layers):
        if src_layer.keys is None or src_layer.values is None:
            continue

        if not dst_layer.is_initialized:
            dst_layer.lazy_initialization(src_layer.keys, src_layer.values)

        if hasattr(dst_layer, "keys") and hasattr(dst_layer, "values"):
            seq_len = src_layer.keys.shape[-2]
            if seq_len > dst_layer.keys.shape[-2]:
                raise ValueError("src cache length exceeds dst cache max length")

            src_batch = src_layer.keys.shape[0]
            copy_batch = src_batch if batch_size is None else min(batch_size, src_batch)

            dst_layer.keys.zero_()
            dst_layer.values.zero_()
            dst_layer.keys[:copy_batch, ..., :seq_len, :].copy_(src_layer.keys[:copy_batch])
            dst_layer.values[:copy_batch, ..., :seq_len, :].copy_(src_layer.values[:copy_batch])
            if hasattr(dst_layer, "cumulative_length"):
                dst_layer.cumulative_length = seq_len


def main(manipulation: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        print(f"Found {device_count} GPU(s)")
        
        # Get handle for first GPU (device 0)
        gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        # Print initial GPU info
        print("\nInitial GPU State:")
        print_gpu_info(gpu_handle)
        
    except Exception as e:
        print(f"Warning: Could not initialize pynvml: {e}")
        gpu_handle = None

    print_gpu_info(gpu_handle)
    print_supported_clocks(gpu_handle)

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
        attn_implementation='flash_attention_2',
    )
    try:
        model.to(device)
    except Exception as e:
        print(f"Error moving model to device: {e}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"  Warning: pad_token was None, set to eos_token ({tokenizer.eos_token})")

    # batch_list = [4]
    test_case = {}

    batch_list = [8]
    # batch_list = [32,16,8,4,2,1]
    # batch_list = [32]
    # extend_batch_list = [b for b in range(16, 0, -1)]
    # batch_list.extend(extend_batch_list)

    graph_batch_list = [8]
    # graph_batch_list = [32,16,8,4,2,1]
    # graph_batch_list = [32]
    # extend_batch_list = [b for b in range(16, 0, -1)]
    # graph_batch_list.extend(extend_batch_list)

    outputs_idx = None
    mask = None
    past_key_values = None
    for batch_size in batch_list:
        info_dict = {
        }
        test_case[batch_size] = info_dict

        print("Process Capture CUDA Graph for batch size:", batch_size)
        
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
        prefill_len = inputs["input_ids"].shape[1]
        cache_position = torch.tensor([prefill_len], device=device, dtype=torch.long)
        outputs_idx = None
        for _ in range(2): 
            with torch.no_grad():
                outputs_idx = model(
                    input_ids=next_token_ids,
                    attention_mask=mask,
                    past_key_values=past_key_values,
                    cache_position=cache_position,
                    use_cache=True,
                    return_dict=True,
                )
        torch.cuda.synchronize()
        # print decoding tokens text
        next_token_logits = outputs_idx.logits[:, -1, :]
        token_ids = torch.argmax(next_token_logits, dim=-1, keepdim=False)

        # print(f"Decoding output token ids: {token_ids}")

        # print decoding tokens text
        
        for i, text in enumerate(token_ids):
            decoded_texts = tokenizer.batch_decode(text, skip_special_tokens=True)
            # print(f"input: {prompts[i]}{prefill_test[i][0]}")
            # print(f"Decoded text for batch {i}: {decoded_texts}")
            prefill_test.append(decoded_texts)
        # print("=" * 60)
        
        # ones = torch.ones_like(next_token_ids).to(device)
        # next_token_ids = torch.add(next_token_ids, ones)
        # print(f"Next token ids for CUDA graph capture: {next_token_ids}")

        if batch_size in graph_batch_list:
            # Allocate static buffers for CUDA Graph replay
            static_input_ids = torch.empty_like(next_token_ids)
            static_attention_mask = torch.empty_like(mask)
            static_cache_position = torch.tensor([prefill_len], device=device, dtype=torch.long)

            src_cache_len = past_key_values.get_seq_length() if hasattr(past_key_values, "get_seq_length") else 0
            max_cache_len = max(inputs["input_ids"].shape[1] + 1, src_cache_len + 1)
            static_past_key_values = transformers.StaticCache(config=model.config, max_cache_len=max_cache_len)
            copy_cache_tensors(static_past_key_values, past_key_values, batch_size=batch_size)

            # Initialize static buffers with current data before capture
            static_input_ids.copy_(next_token_ids)
            static_attention_mask.copy_(mask)

            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph):
                with torch.no_grad():
                    static_outputs = model(
                        input_ids=static_input_ids,
                        attention_mask=static_attention_mask,
                        past_key_values=static_past_key_values,
                        cache_position=static_cache_position,
                        use_cache=True,
                        return_dict=True,
                    )
            
            graph.instantiate()
            graph.replay()
            # print(f"Replayed CUDA graph for batch size {batch_size}")
            torch.cuda.synchronize()
            ## print replay output tokens
            next_token_logits = static_outputs.logits[:, -1, :]
            token_ids = torch.argmax(next_token_logits, dim=-1, keepdim=False)
            print(f"Decoding output token ids after graph replay: {token_ids}")
            
            # for i, text in enumerate(token_ids):
            #     decoded_texts = tokenizer.batch_decode(text, skip_special_tokens=True)
            #     print(f"input: {prompts[i]}{prefill_test[i][0]}")
            #     print(f"Decoded text for batch {i}: {decoded_texts[0]}")
            test_case[batch_size]['graph'] = graph
            test_case[batch_size]['graph_batch_size'] = batch_size
            test_case[batch_size]['outputs'] = token_ids
            test_case[batch_size]['inputs'] = {
                'input_ids': next_token_ids,
                'attention_mask': mask,
                'past_key_values': past_key_values,
            }
            test_case[batch_size]['static_outputs'] = static_outputs
            test_case[batch_size]['static_inputs'] = {
                'input_ids': static_input_ids,
                'attention_mask': static_attention_mask,
                'cache_position': static_cache_position,
                'past_key_values': static_past_key_values,
            }
            test_case[batch_size]['manipulated'] = False
            test_case[batch_size]['manipulation_batch_size'] = batch_size
            graph.enable_debug_mode()

            os.makedirs(f"cuda_graphs", exist_ok=True)
            file_path = f"cuda_graphs/currunt_llama3_8b_bs_{batch_size}_graph.dot"
            graph.debug_dump(file_path)
        else:
            graph_batch = -1
            for i in graph_batch_list:
                if batch_size < i:
                    graph_batch = i
                else:
                    break
            test_case[batch_size]['graph'] = test_case[graph_batch]['graph']
            test_case[batch_size]['graph_batch_size'] = graph_batch
            test_case[batch_size]['outputs'] = token_ids
            test_case[batch_size]['inputs'] = {
                'input_ids': next_token_ids,
                'attention_mask': mask,
                'past_key_values': past_key_values,
            }
            test_case[batch_size]['static_outputs'] = static_outputs
            test_case[batch_size]['static_inputs'] = {
                'input_ids': static_input_ids,
                'attention_mask': static_attention_mask,
                'cache_position': static_cache_position,
                'past_key_values': static_past_key_values,
            }

    for batch_size, values in test_case.items():
        graph_batch_size = values['graph_batch_size']
        decoding_steps = 1
        cudagraph_mode = "FULL"
        print("\n==============================")
        print(f"Captured CUDA graph for batch size {batch_size}")
        print(f" Graph batch size: {graph_batch_size}")
        print(f" input_ids shape: {values['inputs']['input_ids'].shape}")
        print(f" attention_mask shape: {values['inputs']['attention_mask'].shape}")
        print(f" past_key_values length: {len(values['inputs']['past_key_values'])}")
        print(f" outputs shape: {values['outputs'].shape}")
        print("==============================\n")

        
        graph = None
        graph_input_pointer = None
        graph_output_pointer = None
        if batch_size not in graph_batch_list:
            for b in graph_batch_list:
                if batch_size < b:
                    graph_batch = b
                else:
                    break
            graph = test_case[graph_batch]['graph']
            graph_input_pointer = test_case[graph_batch]['static_inputs']
            graph_output_pointer = test_case[graph_batch]['static_outputs']
            if manipulation and batch_size != 3:
                if batch_size != test_case[graph_batch]['manipulation_batch_size']:
                    try:
                        file_path = f"cuda_graphs/manipulate_llama3_8b_bs_{graph_batch}_{batch_size}_graph.dot"
                        raw_capsule = graph.raw_cuda_graph()
                        ret = cuda_graph_api.manipulation_huggingface_graph(raw_capsule, batch_size, file_path, False)
                        graph.instantiate()
                        print(f"Manipulated graph from batch size {graph_batch} to {batch_size}")
                        print(ret)
                        test_case[graph_batch]['manipulated'] = True
                        test_case[graph_batch]['manipulation_batch_size'] = batch_size
                    except Exception as e:
                        print(f"Error manipulating graph: {e}")

        else:
            graph = values['graph']
            graph_input_pointer = values['static_inputs']
            graph_output_pointer = values['static_outputs']

        # try:
        #     file_path = f"cuda_graphs/manipulate_llama3_8b_bs_{graph_batch}_5_graph.dot"
        #     raw_capsule = graph.raw_cuda_graph()
        #     graph.instantiate()
        #     ret = cuda_graph_api.manipulation_huggingface_graph(raw_capsule, 5, file_path)
        #     print(f"Manipulated graph from batch size {graph_batch} to 5")
        #     print(ret)
        #     test_case[graph_batch]['manipulated'] = True
        #     test_case[graph_batch]['manipulation_batch_size'] = batch_size

        # except Exception as e:
        #     print(f"Error manipulating graph: {e}")

        target_batch = graph_input_pointer['input_ids'].shape[0]
        src_batch = values['inputs']['input_ids'].shape[0]
        copy_batch = min(target_batch, src_batch)

        graph_input_pointer['input_ids'].zero_()
        graph_input_pointer['input_ids'][:copy_batch].copy_(values['inputs']['input_ids'][:copy_batch])

        min_dtype = torch.finfo(graph_input_pointer['attention_mask'].dtype).min
        graph_input_pointer['attention_mask'].fill_(min_dtype)
        graph_input_pointer['attention_mask'][:copy_batch].copy_(values['inputs']['attention_mask'][:copy_batch])

        copy_cache_tensors(graph_input_pointer['past_key_values'], values['inputs']['past_key_values'], batch_size=copy_batch)

        monitor_graph = GPUMonitor(device_id=0, batch_size=batch_size, graph_batch_size=graph_batch_size, decoding_steps=decoding_steps, cudagraph_mode=cudagraph_mode, manipulated=manipulation)

        def monitor_batch_graph():
            torch.cuda.synchronize()
            start_time = time.time()
            for _ in range(1):
                graph.replay()
            torch.cuda.synchronize()
            end_time = time.time()
            total_time = end_time - start_time
            return _, total_time

        for _ in range(1):
            _, during_time = monitor_graph.collect_during_execution(
                monitor_batch_graph,
                num_samples = batch_size,
            ) 
            monitor_graph.save_statistics_to_csv(during_time=during_time)
        
        output_ids = graph_output_pointer
        next_token_logits = output_ids.logits[:, -1, :]
        token_ids = torch.argmax(next_token_logits, dim=-1, keepdim=False)
        print(f"Decoding output token ids after graph replay[{len(token_ids)}]: {token_ids}")





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## 인자가 있으면 true 없으면 false    
    # parser.add_argument("--manipulation", type=bool, default=False, help="Manipulation Flag")
    # args = parser.parse_args()
    main(True)