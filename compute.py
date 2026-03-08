import torch
import os
import gc
import sys
from typing import Dict, List
from transformers import AutoConfig, AutoModelForCausalLM
from safetensors.torch import load_file

try:
    import torch_npu
except ImportError:
    pass


class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


MODEL_PATH = "/data/models/huggingface/Qwen/Qwen3-30B-A3B"

if hasattr(torch, "npu") and torch.npu.is_available():
    DEVICE = "npu"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
SEQ_LENGTHS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
WARM_UP_STEPS = 15
TEST_STEPS = 50
TARGET_LAYERS = 5

REQUIRED_SHARDS = [
    "model-00001-of-00016.safetensors",
    "model-00002-of-00016.safetensors",
    "model-00003-of-00016.safetensors",
]

per_layer_time: Dict[int, Dict[str, List[float]]] = {}
module_time_accumulator: Dict[str, List[float]] = {
    "attention": [],
    "moe_router": [],
    "moe_mlp_total": [],  # 整个 MLP 块（含调度和专家）
    "moe_expert": [],  # 单个 Expert 的计算时间
    "layer_norm": [],
    "total_layer": [],
}


def create_hooks(layer_idx: int, module_name: str):
    def pre_hook(module, input):
        start_event = getattr(torch, DEVICE).Event(enable_timing=True)
        end_event = getattr(torch, DEVICE).Event(enable_timing=True)
        start_event.record()
        module._cuda_timer = (start_event, end_event)

    def post_hook(module, input, output):
        start_event, end_event = module._cuda_timer
        end_event.record()
        getattr(torch, DEVICE).synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        elapsed_time_sec = elapsed_time_ms / 1000.0

        module_time_accumulator[module_name].append(elapsed_time_sec)
        per_layer_time[layer_idx][module_name].append(elapsed_time_sec)

    return pre_hook, post_hook


def create_expert_hooks(layer_idx: int, module_name: str):
    def pre_hook(module, input):
        # input[0] 是传入 expert 的 hidden_states
        # 如果当前 expert 没有被分配到 token (numel == 0)，则跳过计时
        if input[0].numel() == 0:
            module._skip_timing = True
            return
        module._skip_timing = False

        start_event = getattr(torch, DEVICE).Event(enable_timing=True)
        end_event = getattr(torch, DEVICE).Event(enable_timing=True)
        start_event.record()
        module._cuda_timer = (start_event, end_event)

    def post_hook(module, input, output):
        if getattr(module, "_skip_timing", False):
            return

        start_event, end_event = module._cuda_timer
        end_event.record()
        getattr(torch, DEVICE).synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
        elapsed_time_sec = elapsed_time_ms / 1000.0

        module_time_accumulator[module_name].append(elapsed_time_sec)
        per_layer_time[layer_idx][module_name].append(elapsed_time_sec)

    return pre_hook, post_hook


def bind_hooks_to_model(model: AutoModelForCausalLM) -> None:
    print(f"正在绑定钩子 (针对 Qwen3 MoE 结构优化)...")
    for layer_idx in range(TARGET_LAYERS):
        per_layer_time[layer_idx] = {
            "attention": [],
            "moe_router": [],
            "moe_mlp_total": [],
            "moe_expert": [],
            "layer_norm": [],
            "total_layer": [],
        }
        layer = model.model.layers[layer_idx]

        # 1. 整层
        layer.register_forward_pre_hook(create_hooks(layer_idx, "total_layer")[0])
        layer.register_forward_hook(create_hooks(layer_idx, "total_layer")[1])

        # 2. Attention
        layer.self_attn.register_forward_pre_hook(
            create_hooks(layer_idx, "attention")[0]
        )
        layer.self_attn.register_forward_hook(create_hooks(layer_idx, "attention")[1])

        # 3. MoE 逻辑
        # 绑定整个 MLP (MoE Block)
        layer.mlp.register_forward_pre_hook(create_hooks(layer_idx, "moe_mlp_total")[0])
        layer.mlp.register_forward_hook(create_hooks(layer_idx, "moe_mlp_total")[1])

        # 寻找并绑定 Router/Gate
        router_module = None
        for name, sub_m in layer.mlp.named_modules():
            if "gate" in name.lower() and "up_proj" not in name.lower():
                router_module = sub_m
                break
        if router_module:
            router_module.register_forward_pre_hook(
                create_hooks(layer_idx, "moe_router")[0]
            )
            router_module.register_forward_hook(
                create_hooks(layer_idx, "moe_router")[1]
            )

        # 绑定单个 Expert (仅在当前层是 MoE 层时生效)
        if hasattr(layer.mlp, "experts"):
            for expert in layer.mlp.experts:
                expert.register_forward_pre_hook(
                    create_expert_hooks(layer_idx, "moe_expert")[0]
                )
                expert.register_forward_hook(
                    create_expert_hooks(layer_idx, "moe_expert")[1]
                )

        # 4. LayerNorm
        layer.input_layernorm.register_forward_pre_hook(
            create_hooks(layer_idx, "layer_norm")[0]
        )
        layer.input_layernorm.register_forward_hook(
            create_hooks(layer_idx, "layer_norm")[1]
        )
        layer.post_attention_layernorm.register_forward_pre_hook(
            create_hooks(layer_idx, "layer_norm")[0]
        )
        layer.post_attention_layernorm.register_forward_hook(
            create_hooks(layer_idx, "layer_norm")[1]
        )


def load_partial_model_shards(model_path, required_shards, dtype, target_layers):
    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
        # attn_implementation="eager",
        attn_implementation="flash_attention_2",
        # attn_implementation="sdpa",
    )
    config.num_hidden_layers = target_layers
    with torch.device(DEVICE):
        model = AutoModelForCausalLM.from_config(
            config, trust_remote_code=True, torch_dtype=dtype
        )
    # print(f"当前模型注意力实现方式: {model.config._attn_implementation}")
    state_dict = {}
    for shard_name in required_shards:
        shard_path = os.path.join(model_path, shard_name)
        state_dict.update(load_file(shard_path, device=DEVICE))
    model.load_state_dict(state_dict, strict=False)
    return model


def run_decode_timing_test(model, batch_size, seq_length):
    print(f"配置: Batch Size = {batch_size}, Seq Length = {seq_length}")

    # 生成随机的 input_ids 模拟 Prefill 阶段的输入
    input_ids = torch.randint(0, 10000, (batch_size, seq_length), device=DEVICE)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        # 准备 Decode 输入 [Batch, 1]（只取最后一个 token）
        decode_input_ids = input_ids[:, -1:]

    # 预热
    for _ in range(WARM_UP_STEPS):
        outputs = model(
            input_ids=decode_input_ids, past_key_values=past_key_values, use_cache=True
        )

    # 清空计时器
    for m in module_time_accumulator:
        module_time_accumulator[m] = []
    for l in per_layer_time:
        for m in per_layer_time[l]:
            per_layer_time[l][m] = []

    # 正式测试
    print(f"正在执行 {TEST_STEPS} 步自回归解码测量...")
    with torch.no_grad():
        for _ in range(TEST_STEPS):
            outputs = model(
                input_ids=decode_input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values

    print("\n" + "=" * 110)
    print(
        f"{'层索引':<6} | {'Attention':<12} | {'Router':<10} | {'单Expert':<10} | {'MoE(Comp+Comm)':<16} | {'LN(x2)':<10} | {'单层总计'}"
    )
    print("-" * 110)

    for i in range(TARGET_LAYERS):
        d = per_layer_time[i]
        to_ms = lambda name: (sum(d[name]) / len(d[name]) * 1000) if d[name] else 0

        # 关键：MoE 专家及调度时间 = 整个 MLP 耗时
        moe_total = to_ms("moe_mlp_total")
        router = to_ms("moe_router")
        single_expert = to_ms("moe_expert")
        # 我们可以通过计算得出“专家计算+调度”的时间
        experts_plus_dispatch = moe_total - router

        print(
            f"L{i:<5} | {to_ms('attention'):>8.4f} ms | {router:>7.4f} ms | {single_expert:>8.4f} ms | {experts_plus_dispatch:>13.4f} ms | {to_ms('layer_norm'):>7.4f} ms | {to_ms('total_layer'):>8.4f} ms"
        )
    print("=" * 110)


if __name__ == "__main__":
    if DEVICE == "npu":
        device_name = torch.npu.get_device_name(0)
    elif DEVICE == "cuda":
        device_name = torch.cuda.get_device_name(0)
    else:
        device_name = "cpu"
    device_name = device_name.replace(" ", "_").replace("/", "_")

    model = load_partial_model_shards(
        MODEL_PATH, REQUIRED_SHARDS, torch.bfloat16, TARGET_LAYERS
    )
    model.eval()

    attn_impl = getattr(model.config, "_attn_implementation", "unknown")
    log_filename = f"{device_name}_{attn_impl}_layers{TARGET_LAYERS}.log"
    sys.stdout = Logger(log_filename)
    print(f"所有输出将同时保存至: {log_filename}\n")

    bind_hooks_to_model(model)

    print("\n" + "*" * 50)
    print("开始批量参数测试")
    print("*" * 50)

    for bs in BATCH_SIZES:
        for seq_len in SEQ_LENGTHS:
            try:
                run_decode_timing_test(model, bs, seq_len)
            except Exception as e:
                if "out of memory" in str(e).lower():
                    print(
                        f"\n[OOM Error] Batch Size = {bs}, Seq Length = {seq_len} 导致显存溢出，跳过此配置。"
                    )
                else:
                    print(
                        f"\n[Error] Batch Size = {bs}, Seq Length = {seq_len} 测试失败: {e}"
                    )

            gc.collect()
            if DEVICE in ["cuda", "npu"]:
                getattr(torch, DEVICE).empty_cache()
