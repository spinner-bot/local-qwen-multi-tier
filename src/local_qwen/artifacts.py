from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .config import MODEL_REGISTRY


def verify_model_dir(model_dir: Path) -> list[str]:
    """Check the download marker and all checkpoint shards referenced by the index."""
    model_dir = Path(model_dir)
    errors: list[str] = []
    if not model_dir.is_dir():
        return [f"目录不存在：{model_dir}"]
    if not (model_dir / ".download_complete").is_file():
        errors.append("缺少 .download_complete 标记")
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        errors.append("缺少 model.safetensors.index.json")
        return errors
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shards = sorted(set(index["weight_map"].values()))
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        errors.append(f"索引文件无效：{exc}")
        return errors
    if not shards:
        errors.append("权重索引没有分片记录")
    for shard in shards:
        candidate = model_dir / shard
        if not candidate.is_file() or candidate.stat().st_size == 0:
            errors.append(f"权重分片缺失或为空：{shard}")
    for required in ("config.json", "tokenizer_config.json"):
        if not (model_dir / required).is_file():
            errors.append(f"缺少文件：{required}")
    return errors


def _registry() -> list[dict[str, str]]:
    with MODEL_REGISTRY.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="校验本地模型权重和分片完整性")
    parser.add_argument("tiers", nargs="*", help="层级，如 l1 l2；省略时检查非 optional 模型")
    args = parser.parse_args(argv)
    rows = _registry()
    selected = {name.lower() for name in args.tiers}
    if selected:
        rows = [row for row in rows if row.get("tier", "").lower() in selected]
        found = {row.get("tier", "").lower() for row in rows}
        missing = selected - found
        if missing:
            parser.error(f"未知层级：{', '.join(sorted(missing))}")
    else:
        rows = [row for row in rows if row.get("status", "").lower() != "optional"]
    failed = False
    for row in rows:
        tier = row.get("tier", "?").lower()
        errors = verify_model_dir(Path(row["local_path"]))
        if errors:
            failed = True
            print(f"FAIL {tier}: " + "；".join(errors))
        else:
            print(f"OK   {tier}: {row['local_path']}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
