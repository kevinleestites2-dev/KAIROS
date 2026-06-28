"""
╔═══════════════════════════════════════════════════════════════╗
║                        K A I R O S                           ║
║           The Right Intelligence at the Right Moment         ║
║                                                               ║
║  Sovereign Multi-Agent Framework — Pantheon Ecosystem         ║
║  Author: Kevin Lee (kevinleestites2-dev)                      ║
║  Stack: Groq → Gemini → DeepSeek → OpenRouter → Ollama       ║
╚═══════════════════════════════════════════════════════════════╝
"""

import json
import os
import re
import subprocess
import time
import traceback
import urllib.parse
import urllib.request
import concurrent.futures
from typing import Any, Optional

# ─────────────────────────────────────────────
# SECTION 1: PROVIDER ROUTER
# ─────────────────────────────────────────────

PROVIDER_CONFIGS = [
    {
        "name": "Groq",
        "env_key": "GROQ_API_KEY",
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "model": "llama3-8b-8192",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
    },
    {
        "name": "Gemini",
        "env_key": "GEMINI_API_KEY",
        "url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "model": "gemini-2.0-flash",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
    },
    {
        "name": "DeepSeek",
        "env_key": "DEEPSEEK_API_KEY",
        "url": "https://api.deepseek.com/chat/completions",
        "model": "deepseek-chat",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
    },
    {
        "name": "OpenRouter",
        "env_key": "OPENROUTER_API_KEY",
        "url": "https://openrouter.ai/api/v1/chat/completions",
        "model": "mistralai/mistral-7b-instruct",
        "auth_header": "Authorization",
        "auth_prefix": "Bearer ",
    },
]

class ProviderRouter:
    def __init__(self):
        self.available = self._detect()

    def _detect(self):
        available = []
        for cfg in PROVIDER_CONFIGS:
            if cfg["env_key"] and os.environ.get(cfg["env_key"]):
                available.append(cfg)
        return available

    def call(self, messages: list, system: str = "", max_tokens: int = 1024) -> str:
        if not self.available:
            # Fallback to a mock response if no keys are found for demonstration
            return "[KAIROS] No LLM providers available. Set API keys to enable LLM thinking."
        
        for provider in self.available:
            try:
                return self._call_provider(provider, messages, system, max_tokens)
            except Exception as e:
                print(f"[ProviderRouter] {provider['name']} failed: {e}. Trying next...")
                continue
        return "[KAIROS] All providers exhausted."

    def _call_provider(self, cfg: dict, messages: list, system: str, max_tokens: int) -> str:
        full_messages = []
        if system:
            full_messages.append({"role": "system", "content": system})
        full_messages.extend(messages)

        payload = json.dumps({
            "model": cfg["model"],
            "messages": full_messages,
            "max_tokens": max_tokens,
            "temperature": 0.7,
        }).encode("utf-8")

        headers = {"Content-Type": "application/json"}
        headers[cfg["auth_header"]] = cfg["auth_prefix"] + os.environ.get(cfg["env_key"], "")

        req = urllib.request.Request(cfg["url"], data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        return data["choices"][0]["message"]["content"].strip()

# ─────────────────────────────────────────────
# SECTION 2: TOOL SUITE
# ─────────────────────────────────────────────

class BaseTool:
    name: str = "base"
    description: str = ""
    def run(self, input: str) -> str:
        raise NotImplementedError

class WebTool(BaseTool):
    name = "web_search"
    description = "Search web via DuckDuckGo. Input: search query."
    def run(self, query: str) -> str:
        try:
            url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json&no_html=1"
            req = urllib.request.Request(url, headers={"User-Agent": "Kairos/1.0"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            result = data.get("AbstractText") or data.get("Definition") or "No direct summary."
            return f"WEB: {result[:200]}"
        except Exception as e: return f"WEB_ERROR: {e}"

class ShellTool(BaseTool):
    name = "shell_exec"
    description = "Execute shell command. Input: command string."
    def run(self, cmd: str) -> str:
        # Integrated IronGate check inside tool for double-safety
        blocked = [r"rm\s+-rf", r"format\s+", r"chmod\s+777"]
        if any(re.search(p, cmd, re.I) for p in blocked): return "SHELL_BLOCKED: Policy Violation."
        try:
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
            return f"SHELL: {(result.stdout or result.stderr).strip()[:300]}"
        except Exception as e: return f"SHELL_ERROR: {e}"

class MemoryTool(BaseTool):
    name = "memory"
    description = "Storage. Input: 'get:key' or 'set:key:value'"
    def __init__(self, store: dict): self.store = store
    def run(self, input: str) -> str:
        if input.startswith("get:"): return f"MEM: {self.store.get(input[4:].strip(), 'N/A')}"
        if input.startswith("set:"):
            parts = input[4:].split(":", 1)
            self.store[parts[0].strip()] = parts[1].strip()
            return f"MEM: {parts[0].strip()} saved."
        return "MEM_ERROR"

# ─────────────────────────────────────────────
# SECTION 3: KAIROS AGENT
# ─────────────────────────────────────────────

class KairosAgent:
    def __init__(self, name, role, goal, backstory, tools=None, provider=None, verbose=True):
        self.name = name
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.tools = {t.name: t for t in (tools or [])}
        self.provider = provider
        self.verbose = verbose

    @property
    def system_prompt(self) -> str:
        t_list = "\n".join([f"- {t.name}: {t.description}" for t in self.tools.values()])
        return f"Name: {self.name}\nRole: {self.role}\nGoal: {self.goal}\nTools:\n{t_list}\nFormat: TOOL: name | INPUT: data"

    def run(self, task: str, context: str = "", bus=None) -> str:
        if self.verbose: print(f"[{self.name}] SENSE: {task[:50]}")
        msg = [{"role": "user", "content": f"Context: {context}\nTask: {task}"}]
        resp = self.provider.call(msg, system=self.system_prompt) if self.provider else "Logic placeholder."
        
        # Tool usage logic
        match = re.search(r"TOOL:\s*(\w+)\s*\|\s*INPUT:\s*(.+)", resp)
        if match:
            t_name, t_in = match.group(1).strip(), match.group(2).strip()
            if t_name in self.tools:
                t_res = self.tools[t_name].run(t_in)
                msg.append({"role": "assistant", "content": resp})
                msg.append({"role": "user", "content": f"Result: {t_res}"})
                resp = self.provider.call(msg, system=self.system_prompt) if self.provider else t_res
        
        if bus: bus.publish(self.name, resp[:100])
        return resp

# ─────────────────────────────────────────────
# SECTION 4: KAIROS BUS & GATE
# ─────────────────────────────────────────────

class KairosBus:
    def publish(self, sender, data): print(f"[BUS] {sender} >> {data}")

class IronGate:
    def audit(self, task: str):
        if re.search(r"rm\s+-rf", task): return False, "Mass deletion"
        return True, "CLEAN"

# ─────────────────────────────────────────────
# SECTION 5: KAIROS CREW
# ─────────────────────────────────────────────

class KairosCrew:
    def __init__(self, name, agents, process="sequential", memory_path="kairos_mem.json"):
        self.name = name
        self.agents = agents
        self.process = process
        self.bus = KairosBus()
        self.gate = IronGate()
        self.memory_path = memory_path

    def kickoff(self, task: str):
        print(f"\n--- {self.name} KICKOFF: {task} ---")
        safe, reason = self.gate.audit(task)
        if not safe: return f"BLOCKED: {reason}"
        
        context = ""
        results = []
        if self.process == "sequential":
            for agent in self.agents:
                res = agent.run(task, context=context, bus=self.bus)
                context = res
                results.append(res)
        return results

# ─────────────────────────────────────────────
# SECTION 6: PANTHEON DEPLOYMENT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    router = ProviderRouter()
    shared_mem = {}
    tools = [WebTool(), ShellTool(), MemoryTool(shared_mem)]

    # The 12-Agent Pantheon using KairosAgent
    pantheon = [
        KairosAgent("IronClaw", "Security Auditor", "Ensure zero-trust compliance", "Hardened gatekeeper", tools, router),
        KairosAgent("ClawMem", "Context Injector", "Provide historical SAFLA data", "Memory specialist", tools, router),
        KairosAgent("TrinityClaw", "Logic Architect", "Decompose complex tasks", "System planner", tools, router),
        KairosAgent("ZeroClaw", "Infra Manager", "Maintain environment stability", "System admin", tools, router),
        KairosAgent("TinyAGI", "Orchestrator", "Coordinate agent handoffs", "Chief of staff", tools, router),
        KairosAgent("OpenBrowserClaw", "Signal Ingester", "Fetch live web data", "Web researcher", tools, router),
        KairosAgent("ClawSwarm", "Parallel Execution", "Scale tasks across threads", "Swarm lead", tools, router),
        KairosAgent("ARC", "Research Synthesizer", "Merge signals into reports", "Deep analyst", tools, router),
        KairosAgent("AutoClaw", "Automation Lead", "Schedule recursive loops", "Task scheduler", tools, router),
        KairosAgent("OpenCrabs", "Performance Lead", "Optimize execution speed", "Rust/Performance", tools, router),
        KairosAgent("PicoClaw", "Edge Strategist", "Optimize for local nodes", "Go/Edge specialist", tools, router),
        KairosAgent("OpenClaw", "Core Executor", "Perform general logic", "Generalist", tools, router),
    ]

    crew = KairosCrew("KAIROS Pantheon", pantheon)
    crew.kickoff("Analyze the Solana ecosystem and report on emerging AI agent trends.")
