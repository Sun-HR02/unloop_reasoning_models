from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import re
import json
import os
import torch
from datetime import datetime
from collections import Counter
from gptqmodel import GPTQModel, QuantizeConfig

MAX_NEW_TOKENS = 20000  
TEMPERATURE = 0.01 # default

# 检测可用设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

print('加载数据 loading datasets:')
ds_2024 = load_dataset("Maxwell-Jia/AIME_2024")
ds_2025 = load_dataset("math-ai/aime25")
print(f"AIME 2024 数据集大小: {len(ds_2024['train'])}")
print(f"AIME 2025 数据集大小: {len(ds_2025['test'])}")
print('加载模型 loading model:')
model_name = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B'

# # pretrain 模型加载
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.float16 if device == "cuda" else torch.float32,  # GPU使用半精度
#     device_map="auto" if device == "cuda" else None,  # 自动分配GPU
# )

# 量化模型加载
c = 'quantized_model/'+model_name.split('/')[-1] + f'_{QUANTIZE_METHOD}_quantized'
model = GPTQModel.load(quantized_model_dir)

if device == "cpu":
    print("警告: 未检测到GPU，使用CPU运行会很慢！")

# 调用模型chat，传入string，model，tokenizer
def chat(message,model,tokenizer):
    # TODO:任务描述，记得step by step
    task_prompt = """

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
        use_cache=True,  # 启用KV缓存加速
    )
    return tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

def chat_quantized(message,model):
    result = model.generate(message)[0] # tokens
    return model.tokenizer.decode(result)

# 提取think内容并返回相关信息
def extract_think_info(output):
    """
    提取think内容并返回相关信息
    支持两种模式：
    1. 完整标签: <think>...</think>
    2. 只有结束标签: 开头到</think>之间的内容
    
    Args:
        output: 模型输出的字符串
    
    Returns:
        dict: {
            'length': think内容的长度，如果没有think标签则返回0,
            'content': think的内容,
            'is_complete': 是否有完整的think标签对,
            'is_truncated': 是否可能被截断（输出达到最大长度且没有结束标签）
        }
    """
    result = {
        'length': 0,
        'content': '',
        'is_complete': False,
        'is_truncated': False
    }
    
    # 模式1: 完整的<think>...</think>标签
    pattern_full = r'<think>(.*?)</think>'
    match = re.search(pattern_full, output, re.DOTALL)
    if match:
        result['length'] = len(match.group(1))
        result['content'] = match.group(1)
        result['is_complete'] = True
        return result
    
    # 模式2: 只有</think>结束标签，取开头到</think>之间的内容
    pattern_end_only = r'^(.*?)</think>'
    match = re.search(pattern_end_only, output, re.DOTALL)
    if match:
        result['length'] = len(match.group(1))
        result['content'] = match.group(1)
        result['is_complete'] = True
        return result
    
    # 模式3: 只有开始标签<think>，可能被截断
    pattern_start_only = r'<think>(.*?)$'
    match = re.search(pattern_start_only, output, re.DOTALL)
    if match:
        result['length'] = len(match.group(1))
        result['content'] = match.group(1)
        result['is_complete'] = False
        result['is_truncated'] = True
        return result
    
    return result

# 判断生成结果是否loop无限循环
def is_loop(output, min_pattern_len=10, max_pattern_len=100, repeat_threshold=3): #TODO: 如果问题的答案就是3333, 会误判吧
    """
    检测输出是否存在循环重复模式
    
    Args:
        output: 模型输出的字符串
        min_pattern_len: 最小重复模式长度
        max_pattern_len: 最大重复模式长度
        repeat_threshold: 重复次数阈值，超过此值认为是循环
    
    Returns:
        bool: 是否存在循环
    """
    if not output or len(output) < min_pattern_len * repeat_threshold:
        return False
    
    # 方法1: 检测连续重复的子串模式
    for pattern_len in range(min_pattern_len, min(max_pattern_len, len(output) // repeat_threshold) + 1):
        # 从输出的后半部分开始检测（循环通常出现在末尾）
        start_pos = len(output) // 2
        for i in range(start_pos, len(output) - pattern_len * repeat_threshold):
            pattern = output[i:i + pattern_len]
            # 检查这个模式是否连续重复出现
            repeat_count = 1
            pos = i + pattern_len
            while pos + pattern_len <= len(output) and output[pos:pos + pattern_len] == pattern:
                repeat_count += 1
                pos += pattern_len
            
            if repeat_count >= repeat_threshold:
                return True
    
    # 方法2: 检测输出末尾是否有大量重复字符
    if len(output) > 50:
        tail = output[-50:]
        most_common_char = max(set(tail), key=tail.count)
        if tail.count(most_common_char) > 40:  # 80%以上是同一字符
            return True
    
    # 方法3: 检测句子/段落级别的重复（针对思考回退循环）
    # 按换行符分割成句子/段落
    lines = [line.strip() for line in output.split('\n') if line.strip()]
    if len(lines) >= repeat_threshold:
        line_counts = Counter(lines)
        # 如果某个句子重复出现超过阈值次数
        for line, count in line_counts.most_common():
            if count >= repeat_threshold and len(line) >= min_pattern_len:
                return True
    
    # 方法4: 检测特定思考模式的高频重复（如 "Wait," 开头的回退思考）
    # 这种模式常见于模型陷入反复思考循环
    think_patterns = [
        r'Wait,\s*',           # "Wait, ..." 模式
        r'But\s+wait,\s*',     # "But wait, ..." 模式
        r'Hmm,\s*',            # "Hmm, ..." 模式
        r'Actually,\s*',       # "Actually, ..." 模式
        r'Let me reconsider',  # 重新考虑模式
        r'No,\s+that\'s not',  # 否定自己模式
    ]
    
    for pattern in think_patterns:
        matches = re.findall(pattern, output, re.IGNORECASE)
        if len(matches) >= 10:  # 如果某个思考模式出现超过10次
            # 进一步检查是否有完全相同的句子重复
            # 提取以该模式开头的完整句子（到换行或句号为止）
            sentence_pattern = pattern + r'[^\n.]*[.\n]?'
            sentences = re.findall(sentence_pattern, output, re.IGNORECASE)
            if len(sentences) >= repeat_threshold:
                sentence_counts = Counter(sentences)
                for sentence, count in sentence_counts.most_common():
                    if count >= repeat_threshold:
                        return True
    
    # 方法5: 检测相似句子的周期性重复（允许轻微差异）
    # 将文本按换行分成段落，检查是否有多个段落高度相似
    paragraphs = [p.strip() for p in output.split('\n\n') if p.strip() and len(p.strip()) > 50]
    if len(paragraphs) >= repeat_threshold * 2:
        # 检查段落之间的相似度
        for i, para1 in enumerate(paragraphs):
            similar_count = 1
            for j, para2 in enumerate(paragraphs):
                if i != j:
                    # 简单相似度检测：前50个字符相同
                    if para1[:50] == para2[:50]:
                        similar_count += 1
            if similar_count >= repeat_threshold:
                return True
    
    return False

# 创建结果保存目录和文件
results_dir = "inference_results"
os.makedirs(results_dir, exist_ok=True)

# 生成带时间戳的文件名
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = os.path.join(results_dir, f"inference_results_{model_name.split('/')[-1]}_{timestamp}.jsonl")
stats_file = os.path.join(results_dir, f"statistics_{model_name.split('/')[-1]}_{timestamp}.json")

print(f"结果将保存到: {results_file}")
print(f"统计信息将保存到: {stats_file}")

# 遍历数据集，将Problem作为message传入chat
loop_count = 0  # 循环生成次数
total_count = 0  # 总处理数量
think_lengths = []  # 记录每次的think长度
truncated_count = 0  # 被截断的think数量
all_results = []  # 保存所有结果

# 合并两个数据集
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
print(f"开始推理...\n")

with open(results_file, 'w', encoding='utf-8') as f:
    for item in tqdm(combined_dataset, desc="Processing"):
        problem = item['problem']
        source = item['source']
        answer = item['answer']
        # response = chat(problem, model, tokenizer)
        response = chat_quantized(problem, model)

        total_count += 1
        
        # 统计think信息
        think_info = extract_think_info(response)
        think_lengths.append(think_info['length'])
        
        if think_info['is_truncated']:
            truncated_count += 1
        
        is_looped = is_loop(response)
        if is_looped:
            loop_count += 1
            print(f"检测到循环生成！(第 {loop_count} 次)")
        
        # 准备保存的结果
        result_item = {
            'index': total_count,
            'source': source,
            'problem': problem,
            'answer': answer,
            'response': response,
            'think_info': think_info,
            'is_loop': is_looped,
            'response_length': len(response),
            'timestamp': datetime.now().isoformat()
        }
        
        # 保存到文件（每行一个JSON对象）
        f.write(json.dumps(result_item, ensure_ascii=False) + '\n')
        f.flush()  # 确保实时写入
        
        all_results.append(result_item)
        
        # print(f"Problem: {problem[:100]}...")  # 只打印前100字符
        # print(f"Response: {response[:50]}")

        print("-" * 50)

# 按数据源统计
aime_2024_count = sum(1 for r in all_results if r.get('source') == 'AIME_2024')
aime_2025_count = sum(1 for r in all_results if r.get('source') == 'AIME_2025')
aime_2024_loop = sum(1 for r in all_results if r.get('source') == 'AIME_2024' and r.get('is_loop'))
aime_2025_loop = sum(1 for r in all_results if r.get('source') == 'AIME_2025' and r.get('is_loop'))

# 计算统计信息
stats = {
    'total_count': total_count,
    'loop_count': loop_count,
    'loop_percentage': loop_count/total_count*100 if total_count > 0 else 0,
    'dataset_stats': {
        'aime_2024': {
            'count': aime_2024_count,
            'loop_count': aime_2024_loop,
            'loop_percentage': aime_2024_loop/aime_2024_count*100 if aime_2024_count > 0 else 0
        },
        'aime_2025': {
            'count': aime_2025_count,
            'loop_count': aime_2025_loop,
            'loop_percentage': aime_2025_loop/aime_2025_count*100 if aime_2025_count > 0 else 0
        }
    },
    'think_stats': {
        'total_length': sum(think_lengths),
        'average_length': sum(think_lengths)/len(think_lengths) if think_lengths else 0,
        'max_length': max(think_lengths) if think_lengths else 0,
        'min_length': min(think_lengths) if think_lengths else 0,
        'no_think_count': think_lengths.count(0),
        'truncated_count': truncated_count,
        'truncated_percentage': truncated_count/total_count*100 if total_count > 0 else 0
    },
    'config': {
        'max_new_tokens': MAX_NEW_TOKENS,
        'temperature': TEMPERATURE
    },
    'timestamp': datetime.now().isoformat()
}

# 保存统计信息到文件
with open(stats_file, 'w', encoding='utf-8') as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

# 输出统计结果
print("\n" + "=" * 50)
print(f"统计结果:")
print(f"总处理数量: {total_count}")
print(f"循环生成次数: {loop_count}")
print(f"循环生成比例: {loop_count/total_count*100:.2f}%" if total_count > 0 else "循环生成比例: N/A")
print("-" * 50)
print(f"数据集分布:")
print(f"AIME 2024: {aime_2024_count} 条 (循环: {aime_2024_loop}, {aime_2024_loop/aime_2024_count*100:.2f}%)" if aime_2024_count > 0 else "AIME 2024: 0 条")
print(f"AIME 2025: {aime_2025_count} 条 (循环: {aime_2025_loop}, {aime_2025_loop/aime_2025_count*100:.2f}%)" if aime_2025_count > 0 else "AIME 2025: 0 条")
print("-" * 50)
print(f"Think长度统计:")
print(f"总think长度: {sum(think_lengths)}")
print(f"平均think长度: {sum(think_lengths)/len(think_lengths):.2f}" if think_lengths else "平均think长度: N/A")
print(f"最大think长度: {max(think_lengths)}" if think_lengths else "最大think长度: N/A")
print(f"最小think长度: {min(think_lengths)}" if think_lengths else "最小think长度: N/A")
print(f"无think输出数量: {think_lengths.count(0)}")
print(f"可能被截断的think数量: {truncated_count}")
print(f"截断比例: {truncated_count/total_count*100:.2f}%" if total_count > 0 else "截断比例: N/A")
print("-" * 50)
print(f"结果已保存到: {results_file}")
print(f"统计信息已保存到: {stats_file}")
print("=" * 50)