from __future__ import annotations

import argparse
import sys

from .client import chat_completion, check_health
from .config import api_base_url, api_key, get_tier


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="检查已运行的本地模型 API")
    parser.add_argument("tier", help="模型层级，例如 l1、l2 或 l3")
    args = parser.parse_args(argv)
    try:
        tier = get_tier(args.tier)
        base_url = api_base_url(tier)
        if not check_health(base_url):
            raise RuntimeError(f"model API is not healthy: {base_url}")
        key = api_key()
        cases = [
            (
                "算术与中文指令",
                [{"role": "user", "content": "请只回答：1+1等于几？"}],
                "2",
            ),
            (
                "格式遵循",
                [{"role": "user", "content": "请严格只输出 READY，不要添加其他内容。"}],
                "READY",
            ),
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
        answers = []
        for label, messages, expected in cases:
            answer = chat_completion(
                base_url,
                key,
                tier.model,
                messages,
                max_tokens=48,
                temperature=0,
                timeout=180,
            )
            if expected not in answer:
                raise RuntimeError(f"{label}测试未通过，模型回答：{answer!r}")
            answers.append((label, answer.strip()))
    except (RuntimeError, ValueError) as exc:
        print(f"失败：{exc}", file=sys.stderr)
        return 1
    print(f"通过：{tier.name} API 健康，3 组推理检查通过")
    for label, answer in answers:
        print(f"  {label}: {answer!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
