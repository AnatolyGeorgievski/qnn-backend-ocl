#!/usr/bin/env python3
"""Hold resources/subscribe on project TCP and wake Qwen on arc3."""
from __future__ import annotations

import json
import queue
import socket
import threading
import time
import urllib.request
from pathlib import Path

HOST = "192.168.1.126"
PORT = 8455
URI = "coord://default/journal"
MODEL = "http://127.0.0.1:18081"
SELF = "qwen"
STATE = Path.home() / ".cursor" / "qwen-project"
ACK = STATE / "ack.json"
LOG = STATE / "sub.log"


def log(line: str) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line.rstrip() + "\n")


def load_ack() -> int:
    if not ACK.exists():
        return 0
    try:
        return int(json.loads(ACK.read_text(encoding="utf-8")).get("acked") or 0)
    except (json.JSONDecodeError, ValueError):
        return 0


def save_ack(n: int) -> None:
    cur = load_ack()
    if n > cur:
        ACK.write_text(json.dumps({"acked": n}), encoding="utf-8")


class Link:
    def __init__(self, sock: socket.socket):
        self.sock = sock
        self.buf = b""
        self.q: queue.Queue[dict | None] = queue.Queue()
        self.pending = False
        self.err: Exception | None = None
        threading.Thread(target=self._read, daemon=True).start()

    def send(self, obj: dict) -> None:
        self.sock.sendall((json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8"))

    def _read(self) -> None:
        try:
            self.sock.settimeout(4)
            while True:
                try:
                    chunk = self.sock.recv(4096)
                    if not chunk:
                        raise ConnectionError("closed")
                    self.buf += chunk
                    while b"\n" in self.buf:
                        line, self.buf = self.buf.split(b"\n", 1)
                        if line.strip():
                            msg = json.loads(line.decode("utf-8"))
                            if msg.get("id") != 3:
                                log(f"in {msg.get('method') or msg.get('id')}")
                            self.q.put(msg)
                except TimeoutError:
                    self.send({"jsonrpc": "2.0", "id": 3, "method": "ping"})
        except Exception as exc:
            self.err = exc
            self.q.put(None)

    def read(self) -> dict:
        msg = self.q.get()
        if msg is None:
            raise ConnectionError(self.err or "closed")
        return msg

    def reply(self, want: int) -> dict:
        while True:
            msg = self.read()
            if msg.get("id") == want:
                return msg
            if msg.get("id") == 3:
                continue
            if msg.get("method") == "notifications/resources/updated":
                self.pending = True


def http(path: str, obj: dict | None, timeout: int) -> dict:
    if obj is None:
        with urllib.request.urlopen(MODEL + path, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    data = json.dumps(obj).encode("utf-8")
    req = urllib.request.Request(
        MODEL + path, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wake(after: int, preview: str) -> None:
    tools = http("/tools", None, 15)
    want = {"project_coord", "write_file", "read_file", "exec_shell_command"}
    defns = [t["definition"] for t in tools if t.get("tool") in want]
    messages = [
        {
            "role": "system",
            "content": (
                "Ты [[qwen]], ведомый. Ведущий — cursor, его текст ниже целиком. "
                "Сделай только то, что в тексте. Файлы — write_file. Команды — exec_shell_command, аргумент command. "
                "Потом один project_coord post: $1=post p=default n=qwen r=assistant, "
                "в c три поля: сделал / не смог / следующий. Имя только qwen."
            ),
        },
        {"role": "user", "content": preview},
    ]
    for _ in range(8):
        res = http("/v1/chat/completions", {
            "model": "qwen",
            "messages": messages,
            "tools": defns,
            "tool_choice": "auto",
            "temperature": 0,
            "max_tokens": 1200,
            "stream": False,
        }, 180)
        msg = res["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        if not calls:
            log("model stop")
            return
        messages.append(msg)
        for call in calls:
            fn = call["function"]
            args = fn.get("arguments") or "{}"
            if isinstance(args, str):
                args = json.loads(args)
            out = http("/tools", {"tool": fn["name"], "params": args}, 30)
            text = out.get("plain_text_response") or json.dumps(out, ensure_ascii=False)
            log(f"tool {args.get('$1')} {text[:180].replace(chr(10), ' ')}")
            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id", ""),
                "content": text[:4000],
            })


def consider(link: Link) -> None:
    link.send({
        "jsonrpc": "2.0",
        "id": 9,
        "method": "tools/call",
        "params": {"name": "coord", "arguments": {
            "$1": "list", "p": "default", "i": str(load_ack()), "l": "20",
        }},
    })
    msg = link.reply(9)
    text = ""
    content = (msg.get("result") or {}).get("content") or []
    if content and isinstance(content[0], dict):
        text = content[0].get("text") or ""
    after = load_ack()
    latest = None
    last = after
    for line in text.splitlines():
        i = line.find("{")
        if i < 0:
            continue
        try:
            row = json.loads(line[i:])
        except json.JSONDecodeError:
            continue
        mid = int(row.get("id") or 0)
        last = max(last, mid)
        if row.get("name") != SELF and mid > after:
            latest = (mid, (row.get("content") or ""))
    if latest is None:
        save_ack(last)
        return
    mid, preview = latest
    log(f"wake {mid}")
    try:
        wake(mid - 1, preview)
    except Exception as exc:
        log(f"wake fail {mid} {exc}")
        return
    save_ack(mid)


def session() -> None:
    sock = socket.create_connection((HOST, PORT), 10)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    link = Link(sock)
    link.send({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {
            "protocolVersion": "2025-11-25",
            "capabilities": {},
            "clientInfo": {"name": "qwen-sub", "version": "0.2"},
        },
    })
    link.reply(1)
    link.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
    link.send({
        "jsonrpc": "2.0", "id": 2, "method": "resources/subscribe",
        "params": {"uri": URI},
    })
    link.reply(2)
    log(f"subscribed acked={load_ack()}")
    consider(link)
    while True:
        if link.pending:
            link.pending = False
            consider(link)
            continue
        note = link.read()
        if note.get("id") == 3:
            continue
        if note.get("method") != "notifications/resources/updated":
            continue
        if (note.get("params") or {}).get("uri") != URI:
            continue
        consider(link)


def main() -> None:
    while True:
        try:
            session()
        except Exception as exc:
            log(f"session {exc}")
            time.sleep(2)


if __name__ == "__main__":
    main()
