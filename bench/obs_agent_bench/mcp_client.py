"""Minimal MCP stdio client: spawn `obs mcp`, initialize, list and call tools.

Usage as a CLI (handy for eyeballing tool output):
    python3 mcp_client.py --obs OBS_BIN --dir DATA TOOL '{"run": "bkk07/small"}'
"""

import json
import subprocess
import sys


class McpClient:
    def __init__(self, cmd):
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.n = 0
        init = self.request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "obs-agent-bench", "version": "0"}})
        self.server_info = init
        self.notify("notifications/initialized", {})

    def notify(self, method, params):
        self.p.stdin.write(json.dumps({"jsonrpc": "2.0", "method": method, "params": params}) + "\n")
        self.p.stdin.flush()

    def request(self, method, params):
        self.n += 1
        self.p.stdin.write(json.dumps({"jsonrpc": "2.0", "id": self.n, "method": method, "params": params}) + "\n")
        self.p.stdin.flush()
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError("MCP server closed the connection")
        msg = json.loads(line)
        if "error" in msg:
            raise RuntimeError(f"MCP error: {msg['error']}")
        return msg["result"]

    def tools(self):
        return self.request("tools/list", {})["tools"]

    def call(self, name, args):
        r = self.request("tools/call", {"name": name, "arguments": args})
        text = "\n".join(c.get("text", "") for c in r.get("content", []))
        return text, bool(r.get("isError"))

    def close(self):
        try:
            self.p.stdin.close()
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("tool")
    ap.add_argument("args", nargs="?", default="{}")
    a = ap.parse_args()
    c = McpClient([a.obs, "mcp", "--dir", a.dir])
    text, err = c.call(a.tool, json.loads(a.args))
    print(("[isError]\n" if err else "") + text)
    c.close()
    sys.exit(0)
