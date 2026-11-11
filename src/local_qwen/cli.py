from __future__ import annotations

import argparse
import sys

from .client import chat_completion, check_health
from .config import api_base_url, api_key, get_tier, memory_db_path
from .store import ConversationStore


HELP = """命令：
  /new                       开始新会话
  /clear                     清除当前会话历史
  /remember 键: 内容         显式保存一条本地记忆
  /forget 键                 删除一条本地记忆
  /memories                  查看本地记忆
  /session                   显示当前会话 ID
  /help                      显示帮助
  /exit                      退出
"""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="连接本地 Qwen 服务进行对话")
    parser.add_argument("tier", nargs="?", help="模型层级，例如 l1、l2 或 l3")
    parser.add_argument("--session", help="继续指定的会话 ID")
    parser.add_argument("--prompt", help="单轮提问后退出")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--history-limit", type=int, default=40, help="每次请求携带的最近消息数")
    parser.add_argument("--list-sessions", action="store_true", help="列出本地会话并退出")
    parser.add_argument("--list-memories", action="store_true", help="列出本地记忆并退出")
    return parser


def _system_message(store: ConversationStore) -> dict[str, str]:
    memories = store.memories()
    content = "你是运行在本机的中文助手。请直接、清楚地回答用户。"
    if memories:
        notes = "\n".join(f"- {item['key']}: {item['value']}" for item in memories)
        content += "\n以下是用户明确保存的本地记忆，请仅在相关时作为背景使用：\n" + notes
    return {"role": "system", "content": content}


def _print_memories(store: ConversationStore) -> None:
    items = store.memories()
    if not items:
        print("本地记忆为空。使用 /remember 键: 内容 添加。")
        return
    for item in items:
        print(f"{item['key']}: {item['value']}")


def _print_sessions(store: ConversationStore) -> None:
    sessions = store.list_sessions()
    if not sessions:
        print("还没有保存的会话。")
        return
    for item in sessions:
        title = item["title"] or "（空会话）"
        print(f"{item['id']}  {item['tier']}  {item['message_count']} 条消息  {title}")


def _answer(
    *, store: ConversationStore, session_id: str, tier: str, base_url: str,
    key: str, text: str, history_limit: int, max_tokens: int, temperature: float,
) -> str:
    messages = [_system_message(store)]
    messages.extend(store.history(session_id, limit=history_limit))
    messages.append({"role": "user", "content": text})
    answer = chat_completion(
        base_url, key, tier, messages,
        max_tokens=max_tokens, temperature=temperature,
    )
    store.append_turn(session_id, text, answer)
    return answer


def _handle_command(line: str, store: ConversationStore, session_id: str) -> tuple[str, str] | None:
    command, _, argument = line.partition(" ")
    argument = argument.strip()
    if command in {"/exit", "/quit"}:
        return "exit", session_id
    if command == "/help":
        print(HELP)
    elif command == "/new":
        session_id = store.create_session(store.session(session_id)["tier"])
        print(f"新会话：{session_id}")
    elif command == "/session":
        print(session_id)
    elif command == "/clear":
        store.clear_history(session_id)
        print("已清除当前会话历史；本地记忆保留。")
    elif command == "/memories":
        _print_memories(store)
    elif command == "/remember":
        key, separator, value = argument.partition(":")
        if not separator or not key.strip() or not value.strip():
            print("用法：/remember 键: 内容")
        else:
            store.remember(key, value)
            print(f"已保存记忆：{key.strip()}")
    elif command == "/forget":
        if not argument:
            print("用法：/forget 键")
        elif store.forget(argument):
            print(f"已删除记忆：{argument}")
        else:
            print(f"没有找到记忆：{argument}")
    else:
        print("未知命令。输入 /help 查看可用命令。")
    return None


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    store = ConversationStore(memory_db_path())
    if args.list_sessions:
        _print_sessions(store)
        return 0
    if args.list_memories:
        _print_memories(store)
        return 0
    if not args.tier:
        _parser().error("需要指定模型层级")
    try:
        tier = get_tier(args.tier)
    except ValueError as exc:
        _parser().error(str(exc))
    session = store.session(args.session) if args.session else None
    if args.session and session is None:
        print(f"找不到会话：{args.session}", file=sys.stderr)
        return 2
    if session and session["tier"] != tier.name:
        print(f"会话属于 {session['tier']}，不能用 {tier.name} 继续。", file=sys.stderr)
        return 2
    session_id = args.session or store.create_session(tier.name)
    base_url = api_base_url(tier)
    if not check_health(base_url):
        print(f"模型服务未就绪：{base_url}。可运行 scripts/llm-chat {tier.name} 按需启动。", file=sys.stderr)
        return 3
    try:
        key = api_key()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.prompt is not None:
        try:
            print(_answer(
                store=store, session_id=session_id, tier=tier.name, base_url=base_url,
                key=key, text=args.prompt, history_limit=args.history_limit,
                max_tokens=args.max_tokens, temperature=args.temperature,
            ))
            print(f"会话：{session_id}", file=sys.stderr)
            return 0
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 4

    print(f"连接 {tier.name}；会话 {session_id}。输入 /help 查看命令。")
    while True:
        try:
            line = input("你> ").strip()
        except EOFError:
            print()
            break
        except KeyboardInterrupt:
            print()
            break
        if not line:
            continue
        if line.startswith("/"):
            result = _handle_command(line, store, session_id)
            if result and result[0] == "exit":
                break
            if result:
                session_id = result[1]
            continue
        try:
            answer = _answer(
                store=store, session_id=session_id, tier=tier.name, base_url=base_url,
                key=key, text=line, history_limit=args.history_limit,
                max_tokens=args.max_tokens, temperature=args.temperature,
            )
            print(f"助手> {answer}")
        except RuntimeError as exc:
            print(f"请求失败：{exc}", file=sys.stderr)
    print(f"会话已保存在本地：{session_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
