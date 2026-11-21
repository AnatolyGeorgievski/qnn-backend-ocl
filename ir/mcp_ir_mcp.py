#!/usr/bin/env python3
"""MCP: ir_say — прямая речь SET/граф/GET для Qwen (и Cursor). mcp 2.x MCPServer."""
from __future__ import annotations

import subprocess
from pathlib import Path

from mcp.server.mcpserver import MCPServer

ROOT = Path(__file__).resolve().parent
HELPER = ROOT / "mcp_ir_say_stdin"

mcp = MCPServer("mcp-ir")


@mcp.tool()
def ir_say(speech: str) -> str:
    """Прямая речь MCP+IR: SET $n(shape) числа; граф ADD/MUL/SUB/DIV/NEG/SILU/RMS_NORM/MUL_MAT; GET / GET.vec. Верни ответ как есть."""
    if not HELPER.exists():
        return "status=fail need mcp_ir_say_stdin next to this script"
    p = subprocess.run(
        [str(HELPER)],
        input=speech.encode("utf-8"),
        capture_output=True,
        timeout=60,
    )
    out = p.stdout.decode("utf-8", errors="replace")
    err = p.stderr.decode("utf-8", errors="replace")
    if p.returncode:
        return f"status=fail rc={p.returncode}\n{out}\n{err}"
    return out if out else "status=ok (empty)"


if __name__ == "__main__":
    mcp.run(transport="stdio")
