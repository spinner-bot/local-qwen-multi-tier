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
        answer = chat_completion(
            base_url,
            api_key(),
            tier.model,
            [{"role": "user", "content": "请只回答：1+1等于几？"}],
            max_tokens=32,
            temperature=0,
            timeout=180,
        )
        if "2" not in answer:
            raise RuntimeError(f"smoke prompt returned an unexpected answer: {answer!r}")
    except (RuntimeError, ValueError) as exc:
        print(f"失败：{exc}", file=sys.stderr)
        return 1
    print(f"通过：{tier.name} API 健康，推理回答：{answer.strip()!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
