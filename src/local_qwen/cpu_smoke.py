from __future__ import annotations

import argparse
import os
import sys

from .artifacts import verify_model_dir
from .config import get_tier


CASES = [
    ("算术与中文指令", [{"role": "user", "content": "请只回答：1+1等于几？"}], "2"),
    ("格式遵循", [{"role": "user", "content": "请严格只输出 READY，不要添加其他内容。"}], "READY"),
    (
        "多轮上下文",
        [
            {"role": "user", "content": "我的代号是琥珀。请简短确认。"},
            {"role": "assistant", "content": "收到。"},
            {"role": "user", "content": "我的代号是什么？只回答代号。"},
        ],
        "琥珀",
    ),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="在 CPU 上验证已下载的 BF16 模型")
    parser.add_argument("tier", choices=("l1", "l2"), help="当前支持 BF16 层级 l1 或 l2")
    args = parser.parse_args(argv)
    tier = get_tier(args.tier)
    if tier.format.lower() != "bf16":
        print(f"CPU 冒烟测试仅支持 BF16，{tier.name} 格式为 {tier.format}", file=sys.stderr)
        return 2
    errors = verify_model_dir(tier.local_path)
    if errors:
        print("权重校验失败：" + "；".join(errors), file=sys.stderr)
        return 2
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoTokenizer

        torch.set_num_threads(int(os.environ.get("LLM_CPU_THREADS", "32")))
        tokenizer = AutoTokenizer.from_pretrained(tier.local_path, local_files_only=True)
        model = AutoModelForImageTextToText.from_pretrained(
            tier.local_path,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            local_files_only=True,
        ).eval()
        print(f"已在 CPU 加载 {tier.name}")
        for label, messages, expected in CASES:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=48, do_sample=False)
            answer = tokenizer.decode(
                output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip()
            if expected not in answer:
                print(f"FAIL {label}: {answer!r}", file=sys.stderr)
                return 1
            print(f"PASS {label}: {answer!r}")
        return 0
    except Exception as exc:
        print(f"CPU 模型加载或推理失败：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
