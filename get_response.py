from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import re
import json
import os
import torch
from datetime import datetime
from gptqmodel import GPTQModel, QuantizeConfig

# ==================== Configuration ====================
MAX_NEW_TOKENS = 20000  
TEMPERATURE = 0.01  # default
MODEL_NAME = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'
QUANTIZE_METHOD = 'gptq'
RESULTS_DIR = "inference_results"


def get_device():
    """Detect and return available device (cuda or cpu)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    if device == "cpu":
        print("警告: 未检测到GPU，使用CPU运行会很慢！")
    return device


def load_datasets():
    """Load AIME 2024 and 2025 datasets."""
    print('加载数据 loading datasets:')
    ds_2024 = load_dataset("Maxwell-Jia/AIME_2024")
    ds_2025 = load_dataset("math-ai/aime25")
    print(f"AIME 2024 数据集大小: {len(ds_2024['train'])}")
    print(f"AIME 2025 数据集大小: {len(ds_2025['test'])}")
    return ds_2024, ds_2025


def load_pretrain_model(model_name, device):
    """Load pretrained model and tokenizer."""
    print('加载模型 loading model:')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    return model, tokenizer


def load_quantized_model(model_name, quantize_method='gptq'):
    """Load quantized model."""
    print('加载模型 loading model:')
    quantized_model_dir = 'quantized_model/' + model_name.split('/')[-1] + f'_{quantize_method}_quantized'
    model = GPTQModel.load(quantized_model_dir)
    return model


def chat(message, model, tokenizer):
    """Chat with pretrained model."""
    # TODO: task description, remember step by step
    task_prompt = """
    Think step by step:
    """

    messages = [
        {"role": "user", "content": message},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        **inputs, 
        max_new_tokens=MAX_NEW_TOKENS, 
        temperature=TEMPERATURE,
        do_sample=True if TEMPERATURE > 0 else False,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)


def chat_quantized(message, model):
    """Chat with quantized model."""
    result = model.generate(
        message,
        max_new_tokens=MAX_NEW_TOKENS, 
        temperature=TEMPERATURE
    )[0]
    return model.tokenizer.decode(result)


def extract_think_info(output):
    """
    Extract think content and return related information.
    
    Supports two patterns:
    1. Complete tags: <think>...</think>
    2. Only end tag: content from beginning to </think>
    
    Args:
        output: Model output string
    
    Returns:
        dict: {
            'length': length of think content, 0 if no think tag,
            'content': think content,
            'is_truncated': whether possibly truncated (output reached max length without end tag)
        }
    """
    result = {
        'length': 0,
        'content': '',
        'is_truncated': False
    }
    
    # Pattern 1: Complete <think>...</think> tags
    pattern_full = r'<think>(.*?)</think>'
    match = re.search(pattern_full, output, re.DOTALL)
    if match:
        result['length'] = len(match.group(1))
        result['content'] = match.group(1)
        return result
    
    # Pattern 2: Only </think> end tag, get content from beginning to </think>
    pattern_end_only = r'^(.*?)</think>'
    match = re.search(pattern_end_only, output, re.DOTALL)
    if match:
        result['length'] = len(match.group(1))
        result['content'] = match.group(1)
        return result
    
    # Pattern 3: Only start tag <think>, possibly truncated
    pattern_start_only = r'<think>(.*?)$'
    match = re.search(pattern_start_only, output, re.DOTALL)
    if match:
        result['length'] = len(match.group(1))
        result['content'] = match.group(1)
        result['is_truncated'] = True
        return result
    
    return result


def prepare_combined_dataset(ds_2024, ds_2025):
    """Combine AIME 2024 and 2025 datasets into a single list."""
    combined_dataset = []
    for item in ds_2024['train']:
        combined_dataset.append({
            'problem': item['Problem'],
            'answer': item['Answer'],
            'source': 'AIME_2024'
        })
    for item in ds_2025['test']:
        combined_dataset.append({
            'problem': item['problem'],
            'answer': item['answer'],
            'source': 'AIME_2025'
        })
    print(f"\n合并后总数据量: {len(combined_dataset)}")
    return combined_dataset


def setup_output_files(model_name, results_dir):
    """Create results directory and generate output file paths."""
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f"inference_results_{model_name.split('/')[-1]}_{timestamp}.jsonl")
    stats_file = os.path.join(results_dir, f"statistics_{model_name.split('/')[-1]}_{timestamp}.json")
    print(f"结果将保存到: {results_file}")
    print(f"统计信息将保存到: {stats_file}")
    return results_file, stats_file


def run_inference(combined_dataset, model, results_file, use_quantized=True, tokenizer=None):
    """
    Run inference on combined dataset and save results.
    
    Args:
        combined_dataset: List of problem items
        model: The model to use
        results_file: Path to save results
        use_quantized: Whether using quantized model
        tokenizer: Tokenizer (only needed for pretrained model)
    
    Returns:
        tuple: (all_results, think_lengths, truncated_count)
    """
    total_count = 0
    think_lengths = []
    truncated_count = 0
    all_results = []

    print(f"开始推理...\n")

    with open(results_file, 'w', encoding='utf-8') as f:
        for item in tqdm(combined_dataset, desc="Processing"):
            problem = item['problem']
            source = item['source']
            answer = item['answer']
            
            # Get response based on model type
            if use_quantized:
                response = chat_quantized(problem, model)
            else:
                response = chat(problem, model, tokenizer)

            total_count += 1
            
            # Collect think info
            think_info = extract_think_info(response)
            think_lengths.append(think_info['length'])
            
            if think_info['is_truncated']:
                truncated_count += 1
            
            # Prepare result item
            result_item = {
                'index': total_count,
                'source': source,
                'problem': problem,
                'answer': answer,
                'response': response,
                'think_info': think_info,
                'response_length': len(response),
                'timestamp': datetime.now().isoformat()
            }
            
            # Save to file (one JSON object per line)
            f.write(json.dumps(result_item, ensure_ascii=False) + '\n')
            f.flush()
            
            all_results.append(result_item)
            
            print("-" * 50)

    return all_results, think_lengths, truncated_count


def calculate_statistics(all_results, think_lengths, truncated_count):
    """Calculate statistics from inference results."""
    total_count = len(all_results)
    
    # Count by data source
    aime_2024_count = sum(1 for r in all_results if r.get('source') == 'AIME_2024')
    aime_2025_count = sum(1 for r in all_results if r.get('source') == 'AIME_2025')

    stats = {
        'total_count': total_count,
        'dataset_stats': {
            'aime_2024': {
                'count': aime_2024_count,
            },
            'aime_2025': {
                'count': aime_2025_count,
            }
        },
        'think_stats': {
            'total_length': sum(think_lengths),
            'average_length': sum(think_lengths) / len(think_lengths) if think_lengths else 0,
            'max_length': max(think_lengths) if think_lengths else 0,
            'min_length': min(think_lengths) if think_lengths else 0,
            'no_think_count': think_lengths.count(0),
            'truncated_count': truncated_count,
            'truncated_percentage': truncated_count / total_count * 100 if total_count > 0 else 0
        },
        'config': {
            'max_new_tokens': MAX_NEW_TOKENS,
            'temperature': TEMPERATURE
        },
        'timestamp': datetime.now().isoformat()
    }
    
    return stats


def save_statistics(stats, stats_file):
    """Save statistics to JSON file."""
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def print_statistics(stats, results_file, stats_file):
    """Print statistics to console."""
    total_count = stats['total_count']
    aime_2024_count = stats['dataset_stats']['aime_2024']['count']
    aime_2025_count = stats['dataset_stats']['aime_2025']['count']
    think_stats = stats['think_stats']

    print("\n" + "=" * 50)
    print(f"统计结果:")
    print(f"总处理数量: {total_count}")
    print("-" * 50)
    print(f"数据集分布:")
    print(f"AIME 2024: {aime_2024_count} 条" if aime_2024_count > 0 else "AIME 2024: 0 条")
    print(f"AIME 2025: {aime_2025_count} 条" if aime_2025_count > 0 else "AIME 2025: 0 条")
    print("-" * 50)
    print(f"Think长度统计:")
    print(f"总think长度: {think_stats['total_length']}")
    print(f"平均think长度: {think_stats['average_length']:.2f}" if think_stats['average_length'] > 0 else "平均think长度: N/A")
    print(f"最大think长度: {think_stats['max_length']}" if think_stats['max_length'] > 0 else "最大think长度: N/A")
    print(f"最小think长度: {think_stats['min_length']}" if total_count > 0 else "最小think长度: N/A")
    print(f"无think输出数量: {think_stats['no_think_count']}")
    print(f"可能被截断的think数量: {think_stats['truncated_count']}")
    print(f"截断比例: {think_stats['truncated_percentage']:.2f}%" if total_count > 0 else "截断比例: N/A")
    print("-" * 50)
    print(f"结果已保存到: {results_file}")
    print(f"统计信息已保存到: {stats_file}")
    print("=" * 50)


def main():
    """Main function to run the inference pipeline."""
    # Initialize device
    device = get_device()
    
    # Load datasets
    ds_2024, ds_2025 = load_datasets()
    
    # Load model (choose one of the following)
    # Option 1: Pretrained model
    model, tokenizer = load_pretrain_model(MODEL_NAME, device)
    use_quantized = False
    
    # Option 2: Quantized model
    # model = load_quantized_model(MODEL_NAME, QUANTIZE_METHOD)
    # use_quantized = True
    # tokenizer = None
    
    # Prepare combined dataset
    combined_dataset = prepare_combined_dataset(ds_2024, ds_2025)
    
    # Setup output files
    results_file, stats_file = setup_output_files(MODEL_NAME, RESULTS_DIR)
    
    # Run inference
    all_results, think_lengths, truncated_count = run_inference(
        combined_dataset, 
        model, 
        results_file, 
        use_quantized=use_quantized, 
        tokenizer=tokenizer
    )
    
    # Calculate and save statistics
    stats = calculate_statistics(all_results, think_lengths, truncated_count)
    save_statistics(stats, stats_file)
    
    # Print statistics
    print_statistics(stats, results_file, stats_file)


if __name__ == "__main__":
    main()