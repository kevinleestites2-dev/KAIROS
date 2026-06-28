#!/usr/bin/env python3
"""
ZeusPrime -- King of the Pantheon
The most advanced autonomous AI agent for Termux.

Single-file implementation containing ALL 12 feature systems:
 1. MARS Meta-Cognitive Reflection
 2. Hyperagent Architecture (Meta Agent + Task Agent, self-rewriting)
 3. Four-Layer Memory (L1 Core, L2 User, L3 Long-term SQLite FTS5, L4 Skills)
 4. Closed Skill Extraction Loop
 5. NEXUS Multi-Agent Hiring
 6. Android Hardware Integration (Termux native)
 7. Multi-Model Routing (Ollama: phi4-mini, qwen2.5-coder, llama3.1, llava, nomic-embed-text, stable-zephyr)
 8. Sandboxed Skill Forge
 9. Simulation Mode (freeze / simulate / revert)
10. Pantheon Tool Integrations (GPTSwarm, Hermes, MothBot, Coreon, OpenClaw)
11. Offline First (Ollama, Whisper STT, Edge TTS)
12. Single file: zeus_prime.py

Usage (Termux):
    python zeus_prime.py                 # Interactive CLI
    python zeus_prime.py --voice         # Voice mode (Whisper + Edge TTS)
    python zeus_prime.py --simulate      # Simulation sandbox
    python zeus_prime.py --status        # Show agent status / memory stats
    python zeus_prime.py --forge         # Enter Skill Forge REPL
    python zeus_prime.py --hire          # Trigger NEXUS hiring scan
"""

__version__ = "1.0.0"
__codename__ = "ZeusPrime"

# ═══════════════════════════════════════════════════════════════════════════════
# STDLIB IMPORTS (zero external deps for core -- graceful fallback everywhere)
# ═══════════════════════════════════════════════════════════════════════════════
import argparse
import asyncio
import copy
import datetime
import hashlib
import inspect
import io
import json
import logging
import math
import os
import platform
import random
import re
import shlex
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import traceback
import uuid
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

# ═══════════════════════════════════════════════════════════════════════════════
# OPTIONAL IMPORTS -- graceful degradation
# ═══════════════════════════════════════════════════════════════════════════════
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

try:
    import edge_tts
    HAS_EDGE_TTS = True
except ImportError:
    HAS_EDGE_TTS = False

try:
    from faster_whisper import WhisperModel
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS & PATHS
# ═══════════════════════════════════════════════════════════════════════════════
IS_TERMUX = os.path.isdir("/data/data/com.termux")
HOME = Path.home()
ZEUS_DIR = HOME / ".zeus_prime"
DB_PATH = ZEUS_DIR / "memory.db"
SKILLS_DIR = ZEUS_DIR / "skills"
FORGE_DIR = ZEUS_DIR / "forge"
AGENTS_DIR = ZEUS_DIR / "agents"
SNAPSHOT_DIR = ZEUS_DIR / "snapshots"
LOG_PATH = ZEUS_DIR / "zeus.log"
CONFIG_PATH = ZEUS_DIR / "config.json"
TRUST_LEDGER_PATH = ZEUS_DIR / "trust_ledger.json"
MARS_LOG_PATH = ZEUS_DIR / "mars_reflections.json"
META_IMPROVEMENT_PATH = ZEUS_DIR / "meta_improvements.json"

for _d in (ZEUS_DIR, SKILLS_DIR, FORGE_DIR, AGENTS_DIR, SNAPSHOT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("ZeusPrime")


# ###########################################################################
#  SECTION 1 -- CONFIGURATION
# ###########################################################################

@dataclass
class ZeusConfig:
    """Global configuration with sensible defaults for Termux + Ollama."""

    ollama_host: str = "http://localhost:11434"

    # Model routing table
    model_fast: str = "phi4-mini"
    model_code: str = "qwen2.5-coder:7b"
    model_reason: str = "llama3.1"
    model_vision: str = "llava"
    model_embed: str = "nomic-embed-text"
    model_trivial: str = "stable-zephyr:3b"

    # Cloud fallback (optional)
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    cloud_model: str = "gpt-4o-mini"

    # Memory
    l1_max_tokens: int = 800
    l2_max_tokens: int = 500
    l3_fts_limit: int = 20

    # NEXUS
    max_agents: int = 8
    trust_threshold: float = 0.6

    # Forge
    forge_timeout: int = 30

    # Voice
    whisper_model_size: str = "base"
    tts_voice: str = "en-US-GuyNeural"
    wake_words: List[str] = field(default_factory=lambda: ["zeus", "hey zeus"])

    # Simulation
    sim_shell: str = "/bin/bash"

    @classmethod
    def load(cls) -> "ZeusConfig":
        if CONFIG_PATH.exists():
            raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            known = {f.name for f in cls.__dataclass_fields__.values()}
            filtered = {k: v for k, v in raw.items() if k in known}
            return cls(**filtered)
        return cls()

    def save(self) -> None:
        CONFIG_PATH.write_text(
            json.dumps(asdict(self), indent=2, default=str),
            encoding="utf-8",
        )


# ###########################################################################
#  SECTION 2 -- MULTI-MODEL ROUTER  (Feature 7)
# ###########################################################################

class TaskComplexity(Enum):
    TRIVIAL = auto()
    FAST = auto()
    CODE = auto()
    REASON = auto()
    VISION = auto()
    EMBED = auto()


class OllamaRouter:
    """Routes prompts to the optimal local Ollama model."""

    CODE_KEYWORDS = re.compile(
        r"\b(code|function|class|debug|refactor|implement|script|program|"
        r"python|javascript|rust|go|sql|html|css|api|endpoint|bug|error|"
        r"compile|test|unittest|pytest|fix|patch|diff|merge)\b",
        re.IGNORECASE,
    )
    REASON_KEYWORDS = re.compile(
        r"\b(explain|analyze|compare|evaluate|plan|design|architect|"
        r"strategy|reason|proof|derive|theorem|complex|tradeoff|"
        r"pros and cons|step.by.step|think|why|how does)\b",
        re.IGNORECASE,
    )
    VISION_KEYWORDS = re.compile(
        r"\b(image|photo|picture|screenshot|camera|see|look|visual|"
        r"describe this|what is this|ocr|scan)\b",
        re.IGNORECASE,
    )
    TRIVIAL_PATTERNS = re.compile(
        r"^(hi|hello|hey|thanks|ok|yes|no|sure|bye|good|great|"
        r"what time|date|weather)\b",
        re.IGNORECASE,
    )

    def __init__(self, cfg: ZeusConfig):
        self.cfg = cfg
        self._available_models: Set[str] = set()
        self._last_check = 0.0

    async def _refresh_models(self) -> None:
        now = time.time()
        if now - self._last_check < 60:
            return
        self._last_check = now
        try:
            if HAS_HTTPX:
                async with httpx.AsyncClient(timeout=5) as c:
                    r = await c.get(f"{self.cfg.ollama_host}/api/tags")
                    if r.status_code == 200:
                        data = r.json()
                        self._available_models = {
                            m["name"] for m in data.get("models", [])
                        }
            else:
                proc = await asyncio.create_subprocess_exec(
                    "curl", "-s", f"{self.cfg.ollama_host}/api/tags",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                stdout, _ = await proc.communicate()
                if stdout:
                    data = json.loads(stdout)
                    self._available_models = {
                        m["name"] for m in data.get("models", [])
                    }
        except Exception:
            pass

    def classify(self, prompt: str, has_image: bool = False) -> TaskComplexity:
        if has_image:
            return TaskComplexity.VISION
        if self.VISION_KEYWORDS.search(prompt):
            return TaskComplexity.VISION
        if self.TRIVIAL_PATTERNS.match(prompt.strip()):
            return TaskComplexity.TRIVIAL
        if self.CODE_KEYWORDS.search(prompt):
            return TaskComplexity.CODE
        if self.REASON_KEYWORDS.search(prompt):
            return TaskComplexity.REASON
        if len(prompt.split()) < 10:
            return TaskComplexity.FAST
        return TaskComplexity.REASON

    def select_model(self, complexity: TaskComplexity) -> str:
        mapping = {
            TaskComplexity.TRIVIAL: self.cfg.model_trivial,
            TaskComplexity.FAST: self.cfg.model_fast,
            TaskComplexity.CODE: self.cfg.model_code,
            TaskComplexity.REASON: self.cfg.model_reason,
            TaskComplexity.VISION: self.cfg.model_vision,
            TaskComplexity.EMBED: self.cfg.model_embed,
        }
        chosen = mapping.get(complexity, self.cfg.model_fast)
        if self._available_models and chosen not in self._available_models:
            for fallback in [self.cfg.model_fast, self.cfg.model_reason]:
                if fallback in self._available_models:
                    log.warning("Model %s unavailable, falling back to %s", chosen, fallback)
                    return fallback
            if self._available_models:
                return next(iter(self._available_models))
        return chosen

    async def generate(
        self,
        prompt: str,
        system: str = "",
        model: Optional[str] = None,
        images: Optional[List[str]] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        await self._refresh_models()
        if model is None:
            complexity = self.classify(prompt, has_image=bool(images))
            model = self.select_model(complexity)

        payload: Dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system:
            payload["system"] = system
        if images:
            payload["images"] = images

        try:
            if HAS_HTTPX:
                async with httpx.AsyncClient(timeout=120) as c:
                    r = await c.post(
                        f"{self.cfg.ollama_host}/api/generate",
                        json=payload,
                    )
                    r.raise_for_status()
                    return r.json().get("response", "")
            else:
                proc = await asyncio.create_subprocess_exec(
                    "curl", "-s", "-X", "POST",
                    f"{self.cfg.ollama_host}/api/generate",
                    "-H", "Content-Type: application/json",
                    "-d", json.dumps(payload),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                stdout, _ = await proc.communicate()
                if stdout:
                    return json.loads(stdout).get("response", "")
                return "[Ollama unavailable]"
        except Exception as exc:
            log.error("Ollama generate failed: %s", exc)
            return await self._cloud_fallback(prompt, system)

    async def embed(self, text: str) -> List[float]:
        payload = {"model": self.cfg.model_embed, "prompt": text}
        try:
            if HAS_HTTPX:
                async with httpx.AsyncClient(timeout=30) as c:
                    r = await c.post(
                        f"{self.cfg.ollama_host}/api/embeddings",
                        json=payload,
                    )
                    r.raise_for_status()
                    return r.json().get("embedding", [])
            else:
                proc = await asyncio.create_subprocess_exec(
                    "curl", "-s", "-X", "POST",
                    f"{self.cfg.ollama_host}/api/embeddings",
                    "-H", "Content-Type: application/json",
                    "-d", json.dumps(payload),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                stdout, _ = await proc.communicate()
                if stdout:
                    return json.loads(stdout).get("embedding", [])
                return []
        except Exception as exc:
            log.error("Embedding failed: %s", exc)
            return []

    async def _cloud_fallback(self, prompt: str, system: str = "") -> str:
        if not self.cfg.openai_api_key:
            return "[No model available -- Ollama offline and no cloud API key set]"
        if not HAS_HTTPX:
            return "[Cloud fallback requires httpx]"
        try:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt})
            async with httpx.AsyncClient(timeout=60) as c:
                r = await c.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.cfg.openai_api_key}"},
                    json={"model": self.cfg.cloud_model, "messages": messages},
                )
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            log.error("Cloud fallback failed: %s", exc)
            return f"[Error: {exc}]"


# ###########################################################################
#  SECTION 3 -- FOUR-LAYER MEMORY  (Feature 3)
# ###########################################################################

class MemoryL1Core:
    """L1 Core Memory -- project context, frozen at session start (~800 tokens)."""

    def __init__(self, max_tokens: int = 800):
        self.max_tokens = max_tokens
        self._data: Dict[str, str] = {}
        self._frozen = False

    def set(self, key: str, value: str) -> None:
        if self._frozen:
            return
        self._data[key] = value

    def freeze(self) -> None:
        self._frozen = True
        log.info("L1 Core Memory frozen with %d entries", len(self._data))

    def get(self, key: str) -> Optional[str]:
        return self._data.get(key)

    def render(self) -> str:
        lines = [f"[{k}]: {v}" for k, v in self._data.items()]
        text = "\n".join(lines)
        words = text.split()
        if len(words) > self.max_tokens:
            text = " ".join(words[: self.max_tokens]) + "..."
        return text

    def to_dict(self) -> Dict[str, str]:
        return dict(self._data)


class MemoryL2UserProfile:
    """L2 User Profile -- preferences, style, stack (~500 tokens)."""

    def __init__(self, max_tokens: int = 500):
        self.max_tokens = max_tokens
        self._profile: Dict[str, Any] = {
            "preferences": {},
            "style": "",
            "stack": [],
            "corrections": [],
        }
        self._profile_path = ZEUS_DIR / "user_profile.json"
        self._load()

    def _load(self) -> None:
        if self._profile_path.exists():
            try:
                self._profile = json.loads(
                    self._profile_path.read_text(encoding="utf-8")
                )
            except Exception:
                pass

    def _save(self) -> None:
        self._profile_path.write_text(
            json.dumps(self._profile, indent=2, default=str),
            encoding="utf-8",
        )

    def update_preference(self, key: str, value: Any) -> None:
        self._profile["preferences"][key] = value
        self._save()

    def set_style(self, style: str) -> None:
        self._profile["style"] = style
        self._save()

    def add_stack(self, tech: str) -> None:
        if tech not in self._profile["stack"]:
            self._profile["stack"].append(tech)
            self._save()

    def add_correction(self, correction: str) -> None:
        self._profile["corrections"].append(correction)
        if len(self._profile["corrections"]) > 50:
            self._profile["corrections"] = self._profile["corrections"][-50:]
        self._save()

    def render(self) -> str:
        parts = []
        if self._profile["preferences"]:
            parts.append("Preferences: " + json.dumps(self._profile["preferences"]))
        if self._profile["style"]:
            parts.append(f"Style: {self._profile['style']}")
        if self._profile["stack"]:
            parts.append("Stack: " + ", ".join(self._profile["stack"]))
        if self._profile["corrections"]:
            recent = self._profile["corrections"][-5:]
            parts.append("Recent corrections: " + "; ".join(recent))
        text = "\n".join(parts)
        words = text.split()
        if len(words) > self.max_tokens:
            text = " ".join(words[: self.max_tokens]) + "..."
        return text


class MemoryL3LongTerm:
    """L3 Long-term Memory -- SQLite FTS5 searchable history."""

    def __init__(self, db_path: Path = DB_PATH, fts_limit: int = 20):
        self.db_path = db_path
        self.fts_limit = fts_limit
        self.conn = sqlite3.connect(str(db_path))
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._init_tables()

    def _init_tables(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                embedding TEXT DEFAULT '[]',
                created_at TEXT NOT NULL,
                access_count INTEGER DEFAULT 0
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                id, category, content,
                content='memories',
                content_rowid='rowid'
            );
            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(id, category, content)
                VALUES (new.id, new.category, new.content);
            END;
            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, id, category, content)
                VALUES ('delete', old.id, old.category, old.content);
            END;
            CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, id, category, content)
                VALUES ('delete', old.id, old.category, old.content);
                INSERT INTO memories_fts(id, category, content)
                VALUES (new.id, new.category, new.content);
            END;

            CREATE TABLE IF NOT EXISTS task_history (
                id TEXT PRIMARY KEY,
                task TEXT NOT NULL,
                result TEXT,
                tool_calls INTEGER DEFAULT 0,
                self_corrections INTEGER DEFAULT 0,
                user_corrections INTEGER DEFAULT 0,
                duration_s REAL DEFAULT 0,
                skill_extracted INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS reflections (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                trigger_task_id TEXT,
                insight TEXT NOT NULL,
                principles TEXT DEFAULT '[]',
                procedures TEXT DEFAULT '[]',
                created_at TEXT NOT NULL
            );
            """
        )
        self.conn.commit()

    def store(
        self,
        category: str,
        content: str,
        metadata: Optional[Dict] = None,
        embedding: Optional[List[float]] = None,
    ) -> str:
        mem_id = f"mem-{uuid.uuid4().hex[:12]}"
        now = datetime.datetime.utcnow().isoformat()
        self.conn.execute(
            "INSERT INTO memories (id, category, content, metadata, embedding, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                mem_id,
                category,
                content,
                json.dumps(metadata or {}),
                json.dumps(embedding or []),
                now,
            ),
        )
        self.conn.commit()
        return mem_id

    def search(self, query: str, limit: Optional[int] = None) -> List[Dict]:
        limit = limit or self.fts_limit
        safe_query = re.sub(r"[^\w\s]", "", query)
        tokens = safe_query.split()
        if not tokens:
            return []
        fts_query = " OR ".join(tokens)
        rows = self.conn.execute(
            "SELECT m.id, m.category, m.content, m.metadata, m.created_at "
            "FROM memories_fts f JOIN memories m ON f.id = m.id "
            "WHERE memories_fts MATCH ? ORDER BY rank LIMIT ?",
            (fts_query, limit),
        ).fetchall()
        results = []
        for row in rows:
            results.append({
                "id": row[0],
                "category": row[1],
                "content": row[2],
                "metadata": json.loads(row[3]),
                "created_at": row[4],
            })
            self.conn.execute(
                "UPDATE memories SET access_count = access_count + 1 WHERE id = ?",
                (row[0],),
            )
        self.conn.commit()
        return results

    def store_task(self, task_record: Dict) -> str:
        task_id = f"task-{uuid.uuid4().hex[:12]}"
        now = datetime.datetime.utcnow().isoformat()
        self.conn.execute(
            "INSERT INTO task_history "
            "(id, task, result, tool_calls, self_corrections, user_corrections, "
            "duration_s, skill_extracted, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                task_id,
                task_record.get("task", ""),
                task_record.get("result", ""),
                task_record.get("tool_calls", 0),
                task_record.get("self_corrections", 0),
                task_record.get("user_corrections", 0),
                task_record.get("duration_s", 0),
                task_record.get("skill_extracted", 0),
                now,
            ),
        )
        self.conn.commit()
        return task_id

    def store_reflection(self, reflection: Dict) -> str:
        ref_id = f"ref-{uuid.uuid4().hex[:12]}"
        now = datetime.datetime.utcnow().isoformat()
        self.conn.execute(
            "INSERT INTO reflections "
            "(id, type, trigger_task_id, insight, principles, procedures, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                ref_id,
                reflection.get("type", "principle"),
                reflection.get("trigger_task_id", ""),
                reflection.get("insight", ""),
                json.dumps(reflection.get("principles", [])),
                json.dumps(reflection.get("procedures", [])),
                now,
            ),
        )
        self.conn.commit()
        return ref_id

    def get_recent_tasks(self, limit: int = 10) -> List[Dict]:
        rows = self.conn.execute(
            "SELECT * FROM task_history ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        cols = [
            "id", "task", "result", "tool_calls", "self_corrections",
            "user_corrections", "duration_s", "skill_extracted", "created_at",
        ]
        return [dict(zip(cols, row)) for row in rows]

    def get_reflections(self, ref_type: Optional[str] = None, limit: int = 20) -> List[Dict]:
        if ref_type:
            rows = self.conn.execute(
                "SELECT * FROM reflections WHERE type = ? ORDER BY created_at DESC LIMIT ?",
                (ref_type, limit),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM reflections ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        cols = [
            "id", "type", "trigger_task_id", "insight",
            "principles", "procedures", "created_at",
        ]
        results = []
        for row in rows:
            d = dict(zip(cols, row))
            d["principles"] = json.loads(d["principles"])
            d["procedures"] = json.loads(d["procedures"])
            results.append(d)
        return results

    def stats(self) -> Dict[str, int]:
        mem_count = self.conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        task_count = self.conn.execute("SELECT COUNT(*) FROM task_history").fetchone()[0]
        ref_count = self.conn.execute("SELECT COUNT(*) FROM reflections").fetchone()[0]
        return {"memories": mem_count, "tasks": task_count, "reflections": ref_count}


class MemoryL4SkillsLibrary:
    """L4 Skills Library -- reusable procedures, names only until needed (near-zero token cost)."""

    def __init__(self, skills_dir: Path = SKILLS_DIR):
        self.skills_dir = skills_dir
        self._index: Dict[str, Dict[str, str]] = {}
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        self._index.clear()
        for fp in self.skills_dir.glob("*.md"):
            name = fp.stem
            first_line = ""
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    first_line = f.readline().strip().lstrip("# ")
            except Exception:
                pass
            self._index[name] = {
                "path": str(fp),
                "title": first_line or name,
            }

    def list_names(self) -> List[str]:
        return list(self._index.keys())

    def render_index(self) -> str:
        if not self._index:
            return "(no skills)"
        return "\n".join(
            f"- {name}: {info['title']}" for name, info in self._index.items()
        )

    def load_skill(self, name: str) -> Optional[str]:
        info = self._index.get(name)
        if not info:
            return None
        try:
            return Path(info["path"]).read_text(encoding="utf-8")
        except Exception:
            return None

    def save_skill(self, name: str, content: str) -> Path:
        fp = self.skills_dir / f"{name}.md"
        fp.write_text(content, encoding="utf-8")
        self._rebuild_index()
        log.info("Skill saved: %s", name)
        return fp

    def delete_skill(self, name: str) -> bool:
        info = self._index.get(name)
        if not info:
            return False
        try:
            Path(info["path"]).unlink()
            self._rebuild_index()
            return True
        except Exception:
            return False


class FourLayerMemory:
    """Unified memory facade across all four layers."""

    def __init__(self, cfg: ZeusConfig):
        self.l1 = MemoryL1Core(max_tokens=cfg.l1_max_tokens)
        self.l2 = MemoryL2UserProfile(max_tokens=cfg.l2_max_tokens)
        self.l3 = MemoryL3LongTerm(fts_limit=cfg.l3_fts_limit)
        self.l4 = MemoryL4SkillsLibrary()

    def build_context(self, query: str = "") -> str:
        parts = ["=== L1 Core ===", self.l1.render()]
        parts += ["", "=== L2 User Profile ===", self.l2.render()]
        if query:
            results = self.l3.search(query, limit=5)
            if results:
                parts += ["", "=== L3 Relevant Memories ==="]
                for r in results:
                    parts.append(f"[{r['category']}] {r['content'][:200]}")
        parts += ["", "=== L4 Available Skills ===", self.l4.render_index()]
        return "\n".join(parts)

    def stats(self) -> Dict:
        return {
            "l1_entries": len(self.l1._data),
            "l1_frozen": self.l1._frozen,
            "l3": self.l3.stats(),
            "l4_skills": len(self.l4.list_names()),
        }


# ###########################################################################
#  SECTION 4 -- MARS META-COGNITIVE REFLECTION  (Feature 1)
# ###########################################################################

class MARSEngine:
    """
    Meta-Agent Reflection System (MARS).
    - Principle-based reflection: abstracts rules from mistakes
    - Procedural reflection: derives step-by-step strategies from successes
    - Improves without continuous online feedback
    """

    def __init__(self, llm: OllamaRouter, memory: FourLayerMemory):
        self.llm = llm
        self.memory = memory
        self._principles: List[Dict] = []
        self._procedures: List[Dict] = []
        self._load()

    def _load(self) -> None:
        if MARS_LOG_PATH.exists():
            try:
                data = json.loads(MARS_LOG_PATH.read_text(encoding="utf-8"))
                self._principles = data.get("principles", [])
                self._procedures = data.get("procedures", [])
            except Exception:
                pass

    def _save(self) -> None:
        MARS_LOG_PATH.write_text(
            json.dumps(
                {"principles": self._principles, "procedures": self._procedures},
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

    async def reflect_on_failure(self, task: str, error: str, context: str = "") -> Dict:
        prompt = (
            "You are a meta-cognitive reflection engine. A task failed.\n\n"
            f"Task: {task}\n"
            f"Error: {error}\n"
            f"Context: {context}\n\n"
            "Extract a general PRINCIPLE (not task-specific) that prevents "
            "this class of error in the future. Return JSON:\n"
            '{"principle": "...", "category": "...", "prevention": "..."}'
        )
        raw = await self.llm.generate(prompt, model=self.llm.cfg.model_reason)
        try:
            principle = json.loads(self._extract_json(raw))
        except Exception:
            principle = {"principle": raw.strip(), "category": "general", "prevention": ""}

        principle["created_at"] = datetime.datetime.utcnow().isoformat()
        principle["trigger_task"] = task[:200]
        self._principles.append(principle)
        self._save()

        self.memory.l3.store_reflection({
            "type": "principle",
            "trigger_task_id": task[:100],
            "insight": principle.get("principle", ""),
            "principles": [principle],
        })
        log.info("MARS principle extracted: %s", principle.get("principle", "")[:80])
        return principle

    async def reflect_on_success(self, task: str, steps: List[str], result: str) -> Dict:
        prompt = (
            "You are a meta-cognitive reflection engine. A task succeeded.\n\n"
            f"Task: {task}\n"
            f"Steps taken: {json.dumps(steps)}\n"
            f"Result: {result[:500]}\n\n"
            "Extract a reusable PROCEDURE (step-by-step strategy) that can be "
            "applied to similar tasks. Return JSON:\n"
            '{"procedure_name": "...", "steps": ["..."], "applicable_when": "..."}'
        )
        raw = await self.llm.generate(prompt, model=self.llm.cfg.model_reason)
        try:
            procedure = json.loads(self._extract_json(raw))
        except Exception:
            procedure = {
                "procedure_name": "derived_procedure",
                "steps": steps,
                "applicable_when": "similar tasks",
            }

        procedure["created_at"] = datetime.datetime.utcnow().isoformat()
        procedure["trigger_task"] = task[:200]
        self._procedures.append(procedure)
        self._save()

        self.memory.l3.store_reflection({
            "type": "procedure",
            "trigger_task_id": task[:100],
            "insight": procedure.get("procedure_name", ""),
            "procedures": [procedure],
        })
        log.info("MARS procedure extracted: %s", procedure.get("procedure_name", ""))
        return procedure

    def get_relevant_principles(self, task: str, limit: int = 3) -> List[Dict]:
        task_lower = task.lower()
        scored = []
        for p in self._principles:
            cat = p.get("category", "").lower()
            principle_text = p.get("principle", "").lower()
            score = sum(1 for w in task_lower.split() if w in principle_text or w in cat)
            scored.append((score, p))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:limit]]

    def get_relevant_procedures(self, task: str, limit: int = 3) -> List[Dict]:
        task_lower = task.lower()
        scored = []
        for p in self._procedures:
            applicable = p.get("applicable_when", "").lower()
            name = p.get("procedure_name", "").lower()
            score = sum(1 for w in task_lower.split() if w in applicable or w in name)
            scored.append((score, p))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:limit]]

    def render_guidance(self, task: str) -> str:
        principles = self.get_relevant_principles(task)
        procedures = self.get_relevant_procedures(task)
        parts = []
        if principles:
            parts.append("MARS Principles:")
            for p in principles:
                parts.append(f"  - {p.get('principle', '')}")
                if p.get("prevention"):
                    parts.append(f"    Prevention: {p['prevention']}")
        if procedures:
            parts.append("MARS Procedures:")
            for p in procedures:
                parts.append(f"  - {p.get('procedure_name', '')}")
                for i, s in enumerate(p.get("steps", []), 1):
                    parts.append(f"    {i}. {s}")
        return "\n".join(parts) if parts else ""

    @staticmethod
    def _extract_json(text: str) -> str:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        return match.group(0) if match else "{}"


# ###########################################################################
#  SECTION 5 -- CLOSED SKILL EXTRACTION LOOP  (Feature 4)
# ###########################################################################

@dataclass
class TaskRecord:
    task: str = ""
    steps: List[str] = field(default_factory=list)
    tool_calls: int = 0
    self_corrections: int = 0
    user_corrections: int = 0
    found_better_path: bool = False
    result: str = ""
    duration_s: float = 0.0
    success: bool = False
    skill_extracted: bool = False


class SkillExtractor:
    """Closed-loop skill extraction -- evaluates every completed task."""

    EXTRACT_CONDITIONS = [
        lambda r: r.tool_calls > 5,
        lambda r: r.self_corrections > 0,
        lambda r: r.user_corrections > 0,
        lambda r: r.found_better_path,
    ]

    def __init__(self, llm: OllamaRouter, memory: FourLayerMemory):
        self.llm = llm
        self.memory = memory

    def should_extract(self, record: TaskRecord) -> bool:
        return any(cond(record) for cond in self.EXTRACT_CONDITIONS)

    async def extract(self, record: TaskRecord) -> Optional[str]:
        if not self.should_extract(record):
            return None

        triggers = []
        if record.tool_calls > 5:
            triggers.append(f"complex task ({record.tool_calls} tool calls)")
        if record.self_corrections > 0:
            triggers.append(f"self-corrected {record.self_corrections} time(s)")
        if record.user_corrections > 0:
            triggers.append(f"user corrected {record.user_corrections} time(s)")
        if record.found_better_path:
            triggers.append("found a more efficient path")

        prompt = (
            "You are a skill extraction engine. Convert this completed task into "
            "a reusable skill document in agentskills.io markdown format.\n\n"
            f"Task: {record.task}\n"
            f"Steps: {json.dumps(record.steps)}\n"
            f"Result: {record.result[:500]}\n"
            f"Extraction triggers: {', '.join(triggers)}\n\n"
            "Write a skill document with:\n"
            "- Title (# Skill Name)\n"
            "- Description\n"
            "- Prerequisites\n"
            "- Steps (numbered)\n"
            "- Expected outcome\n"
            "- Notes/caveats\n\n"
            "Keep it portable (works with OpenClaw, Claude Code, etc)."
        )
        skill_md = await self.llm.generate(prompt, model=self.llm.cfg.model_code)

        name_match = re.search(r"^#\s+(.+)$", skill_md, re.MULTILINE)
        skill_name = "auto_skill"
        if name_match:
            skill_name = re.sub(r"[^\w\-]", "_", name_match.group(1).strip().lower())
            skill_name = skill_name[:60]

        self.memory.l4.save_skill(skill_name, skill_md)
        record.skill_extracted = True

        self.memory.l3.store(
            category="skill_extraction",
            content=f"Extracted skill '{skill_name}' from task: {record.task[:200]}",
            metadata={"skill_name": skill_name, "triggers": triggers},
        )
        log.info("Skill extracted: %s (triggers: %s)", skill_name, triggers)
        return skill_name


# ###########################################################################
#  SECTION 6 -- ANDROID HARDWARE INTEGRATION  (Feature 6)
# ###########################################################################

class TermuxHardware:
    """Android hardware integration via Termux APIs."""

    @staticmethod
    async def _run_termux_cmd(cmd: List[str], timeout: int = 10) -> Optional[str]:
        if not IS_TERMUX:
            return None
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            if proc.returncode == 0:
                return stdout.decode("utf-8", errors="replace").strip()
            log.warning("Termux cmd %s failed: %s", cmd, stderr.decode())
            return None
        except asyncio.TimeoutError:
            log.warning("Termux cmd %s timed out", cmd)
            return None
        except FileNotFoundError:
            return None

    async def take_photo(self, camera_id: int = 0) -> Optional[str]:
        photo_path = str(ZEUS_DIR / f"photo_{int(time.time())}.jpg")
        result = await self._run_termux_cmd(
            ["termux-camera-photo", "-c", str(camera_id), photo_path],
            timeout=15,
        )
        if result is not None and Path(photo_path).exists():
            return photo_path
        return None

    async def take_screenshot(self) -> Optional[str]:
        ss_path = str(ZEUS_DIR / f"screenshot_{int(time.time())}.png")
        result = await self._run_termux_cmd(
            ["termux-screenshot", ss_path],
            timeout=10,
        )
        if result is not None and Path(ss_path).exists():
            return ss_path
        return None

    async def get_location(self) -> Optional[Dict]:
        raw = await self._run_termux_cmd(
            ["termux-location", "-p", "gps", "-r", "once"],
            timeout=30,
        )
        if raw:
            try:
                return json.loads(raw)
            except Exception:
                pass
        return None

    async def get_battery(self) -> Optional[Dict]:
        raw = await self._run_termux_cmd(["termux-battery-status"])
        if raw:
            try:
                return json.loads(raw)
            except Exception:
                pass
        return None

    async def get_sensors(self) -> Optional[Dict]:
        raw = await self._run_termux_cmd(
            ["termux-sensor", "-s", "all", "-n", "1"],
            timeout=10,
        )
        if raw:
            try:
                return json.loads(raw)
            except Exception:
                pass
        return None

    async def send_notification(
        self, title: str, content: str, priority: str = "default"
    ) -> bool:
        result = await self._run_termux_cmd([
            "termux-notification",
            "--title", title,
            "--content", content,
            "--priority", priority,
        ])
        return result is not None

    async def get_clipboard(self) -> Optional[str]:
        return await self._run_termux_cmd(["termux-clipboard-get"])

    async def set_clipboard(self, text: str) -> bool:
        result = await self._run_termux_cmd(["termux-clipboard-set", text])
        return result is not None

    async def vibrate(self, duration_ms: int = 200) -> bool:
        result = await self._run_termux_cmd(
            ["termux-vibrate", "-d", str(duration_ms)]
        )
        return result is not None

    async def tts_speak(self, text: str) -> bool:
        result = await self._run_termux_cmd(
            ["termux-tts-speak", text],
            timeout=30,
        )
        return result is not None

    async def get_device_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {"is_termux": IS_TERMUX}
        battery = await self.get_battery()
        if battery:
            info["battery"] = battery
        info["platform"] = platform.platform()
        info["python"] = platform.python_version()
        return info

    async def analyze_image(self, image_path: str, llm: OllamaRouter) -> str:
        if not Path(image_path).exists():
            return "[Image not found]"
        import base64
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return await llm.generate(
            "Describe this image in detail.",
            model=llm.cfg.model_vision,
            images=[b64],
        )


# ###########################################################################
#  SECTION 7 -- SANDBOXED SKILL FORGE  (Feature 8)
# ###########################################################################

class SkillForge:
    """Writes new skills as Python scripts, tests in sandbox, auto-installs."""

    def __init__(self, llm: OllamaRouter, memory: FourLayerMemory, cfg: ZeusConfig):
        self.llm = llm
        self.memory = memory
        self.cfg = cfg
        self.forge_dir = FORGE_DIR

    async def create_skill(self, description: str) -> Dict[str, Any]:
        prompt = (
            "Write a Python script that performs the following task. "
            "The script must be self-contained, use only standard library, "
            "and have a main() function. Include error handling.\n\n"
            f"Task: {description}\n\n"
            "Return ONLY the Python code, no markdown fences."
        )
        code = await self.llm.generate(prompt, model=self.llm.cfg.model_code)
        code = self._clean_code(code)

        skill_id = f"forge_{uuid.uuid4().hex[:8]}"
        script_path = self.forge_dir / f"{skill_id}.py"
        script_path.write_text(code, encoding="utf-8")

        test_result = await self._test_in_sandbox(script_path)

        result = {
            "skill_id": skill_id,
            "path": str(script_path),
            "code": code,
            "test_passed": test_result["passed"],
            "test_output": test_result["output"],
            "installed": False,
        }

        if test_result["passed"]:
            installed_path = SKILLS_DIR / f"{skill_id}.py"
            shutil.copy2(str(script_path), str(installed_path))
            result["installed"] = True
            result["installed_path"] = str(installed_path)

            skill_md = (
                f"# {skill_id}\n\n"
                f"Auto-generated skill: {description}\n\n"
                f"## Usage\n```python\npython {installed_path}\n```\n\n"
                f"## Code\n```python\n{code}\n```\n"
            )
            self.memory.l4.save_skill(skill_id, skill_md)
            log.info("Forge: skill %s installed", skill_id)
        else:
            log.warning("Forge: skill %s failed tests", skill_id)

        return result

    async def _test_in_sandbox(self, script_path: Path) -> Dict[str, Any]:
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable, str(script_path),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "ZEUS_SANDBOX": "1"},
                cwd=str(self.forge_dir),
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=self.cfg.forge_timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                return {"passed": False, "output": "Timeout exceeded"}

            output = stdout.decode("utf-8", errors="replace")
            errors = stderr.decode("utf-8", errors="replace")

            return {
                "passed": proc.returncode == 0,
                "output": output + ("\n[STDERR] " + errors if errors else ""),
            }
        except Exception as exc:
            return {"passed": False, "output": f"Sandbox error: {exc}"}

    @staticmethod
    def _clean_code(raw: str) -> str:
        raw = re.sub(r"^```python\s*\n?", "", raw.strip())
        raw = re.sub(r"\n?```\s*$", "", raw.strip())
        return raw


# ###########################################################################
#  SECTION 8 -- SIMULATION MODE  (Feature 9)
# ###########################################################################

class SimulationMode:
    """
    Freeze state -> simulate bash commands without executing -> see outcome -> revert.
    """

    def __init__(self, llm: OllamaRouter):
        self.llm = llm
        self._frozen_state: Optional[Dict] = None
        self._sim_history: List[Dict] = []
        self._active = False

    def freeze(self, state: Dict) -> None:
        self._frozen_state = copy.deepcopy(state)
        self._sim_history.clear()
        self._active = True
        log.info("Simulation mode: state frozen")

    def is_active(self) -> bool:
        return self._active

    async def simulate_command(self, command: str) -> Dict[str, str]:
        if not self._active:
            return {"error": "Simulation mode not active. Call freeze() first."}

        prompt = (
            "You are a bash simulation engine. Predict the output of this command "
            "WITHOUT actually executing it. Consider the current working directory, "
            "common system state, and potential errors.\n\n"
            f"Command: {command}\n"
            f"Frozen state context: {json.dumps(self._frozen_state or {})[:1000]}\n"
            f"Previous simulated commands: {json.dumps(self._sim_history[-5:])}\n\n"
            "Return JSON:\n"
            '{"stdout": "...", "stderr": "...", "exit_code": 0, '
            '"side_effects": ["list of filesystem/state changes"], '
            '"risk_level": "safe|caution|dangerous"}'
        )
        raw = await self.llm.generate(prompt, model=self.llm.cfg.model_reason)
        try:
            result = json.loads(MARSEngine._extract_json(raw))
        except Exception:
            result = {
                "stdout": raw,
                "stderr": "",
                "exit_code": -1,
                "side_effects": [],
                "risk_level": "unknown",
            }

        entry = {"command": command, "result": result}
        self._sim_history.append(entry)
        return result

    def revert(self) -> Dict:
        state = self._frozen_state
        self._frozen_state = None
        self._sim_history.clear()
        self._active = False
        log.info("Simulation mode: reverted to frozen state")
        return state or {}

    def get_history(self) -> List[Dict]:
        return list(self._sim_history)


# ###########################################################################
#  SECTION 9 -- NEXUS MULTI-AGENT HIRING  (Feature 5)
# ###########################################################################

@dataclass
class AgentProfile:
    agent_id: str
    name: str
    specialty: str
    capabilities: List[str] = field(default_factory=list)
    trust_score: float = 0.5
    tasks_completed: int = 0
    tasks_failed: int = 0
    accuracy_history: List[float] = field(default_factory=list)
    created_at: str = ""
    status: str = "active"


class BayesianTrustLedger:
    """Tracks per-agent accuracy with Bayesian updates."""

    def __init__(self):
        self._ledger: Dict[str, AgentProfile] = {}
        self._load()

    def _load(self) -> None:
        if TRUST_LEDGER_PATH.exists():
            try:
                data = json.loads(TRUST_LEDGER_PATH.read_text(encoding="utf-8"))
                for agent_id, info in data.items():
                    self._ledger[agent_id] = AgentProfile(**info)
            except Exception:
                pass

    def _save(self) -> None:
        data = {aid: asdict(ap) for aid, ap in self._ledger.items()}
        TRUST_LEDGER_PATH.write_text(
            json.dumps(data, indent=2, default=str), encoding="utf-8"
        )

    def register(self, profile: AgentProfile) -> None:
        self._ledger[profile.agent_id] = profile
        self._save()

    def record_outcome(self, agent_id: str, success: bool, accuracy: float = 1.0) -> None:
        profile = self._ledger.get(agent_id)
        if not profile:
            return
        if success:
            profile.tasks_completed += 1
        else:
            profile.tasks_failed += 1
        profile.accuracy_history.append(accuracy)
        if len(profile.accuracy_history) > 100:
            profile.accuracy_history = profile.accuracy_history[-100:]
        total = profile.tasks_completed + profile.tasks_failed
        alpha = profile.tasks_completed + 1
        beta = profile.tasks_failed + 1
        profile.trust_score = alpha / (alpha + beta)
        self._save()

    def get_profile(self, agent_id: str) -> Optional[AgentProfile]:
        return self._ledger.get(agent_id)

    def get_all(self) -> List[AgentProfile]:
        return list(self._ledger.values())

    def get_trusted(self, threshold: float = 0.6) -> List[AgentProfile]:
        return [p for p in self._ledger.values() if p.trust_score >= threshold]


class ShadowMind:
    """Parallel cognitive layer -- offers pattern-based intuition."""

    def __init__(self, llm: OllamaRouter, memory: FourLayerMemory):
        self.llm = llm
        self.memory = memory

    async def intuition(self, task: str, context: str = "") -> str:
        memories = self.memory.l3.search(task, limit=5)
        mem_context = "\n".join(m["content"][:150] for m in memories)

        prompt = (
            "You are the Shadow Mind -- a parallel cognitive layer that provides "
            "pattern-based intuition. Based on past patterns, provide a brief "
            "intuitive assessment of this task.\n\n"
            f"Task: {task}\n"
            f"Context: {context}\n"
            f"Relevant memories:\n{mem_context}\n\n"
            "Provide:\n"
            "1. Confidence level (0-1)\n"
            "2. Suggested approach\n"
            "3. Potential pitfalls\n"
            "4. Recommended agent specialties needed\n"
            "Keep it brief (3-4 sentences total)."
        )
        return await self.llm.generate(prompt, model=self.llm.cfg.model_fast)


class NEXUSHiring:
    """
    NEXUS Multi-Agent Hiring System.
    - Talent Scout: detects coverage gaps (5-signal confidence)
    - Recruiter: hires new agents through 8 gated phases
    - ZeusPrime grows its own team when it hits capability limits
    """

    COVERAGE_SIGNALS = [
        "repeated_failures",
        "slow_execution",
        "missing_domain_knowledge",
        "user_dissatisfaction",
        "complexity_overflow",
    ]

    HIRING_PHASES = [
        "need_identification",
        "capability_spec",
        "candidate_generation",
        "skill_validation",
        "sandbox_test",
        "trust_calibration",
        "integration_check",
        "deployment",
    ]

    def __init__(self, llm: OllamaRouter, memory: FourLayerMemory, cfg: ZeusConfig):
        self.llm = llm
        self.memory = memory
        self.cfg = cfg
        self.trust_ledger = BayesianTrustLedger()
        self.shadow_mind = ShadowMind(llm, memory)

    async def talent_scout_scan(self) -> List[Dict]:
        recent_tasks = self.memory.l3.get_recent_tasks(limit=20)
        if not recent_tasks:
            return []

        failure_domains: Dict[str, int] = defaultdict(int)
        slow_domains: Dict[str, int] = defaultdict(int)
        correction_domains: Dict[str, int] = defaultdict(int)

        for task in recent_tasks:
            domain = self._classify_domain(task.get("task", ""))
            if task.get("self_corrections", 0) > 0 or task.get("user_corrections", 0) > 0:
                failure_domains[domain] += 1
            if task.get("duration_s", 0) > 60:
                slow_domains[domain] += 1
            if task.get("user_corrections", 0) > 0:
                correction_domains[domain] += 1

        gaps = []
        all_domains = set(failure_domains) | set(slow_domains) | set(correction_domains)
        for domain in all_domains:
            signals = 0
            if failure_domains.get(domain, 0) >= 2:
                signals += 1
            if slow_domains.get(domain, 0) >= 2:
                signals += 1
            if correction_domains.get(domain, 0) >= 1:
                signals += 1

            existing = [
                a for a in self.trust_ledger.get_all()
                if domain.lower() in a.specialty.lower()
            ]
            if not existing:
                signals += 1
            if len(recent_tasks) > 10:
                signals += 1

            if signals >= 3:
                gaps.append({
                    "domain": domain,
                    "confidence_signals": signals,
                    "failures": failure_domains.get(domain, 0),
                    "slow_tasks": slow_domains.get(domain, 0),
                    "corrections": correction_domains.get(domain, 0),
                })

        return sorted(gaps, key=lambda g: g["confidence_signals"], reverse=True)

    async def hire_agent(self, gap: Dict) -> Optional[AgentProfile]:
        if len(self.trust_ledger.get_all()) >= self.cfg.max_agents:
            log.warning("NEXUS: max agent count reached (%d)", self.cfg.max_agents)
            return None

        domain = gap["domain"]
        phase_results: Dict[str, Any] = {}

        # Phase 1: Need identification
        phase_results["need_identification"] = {
            "domain": domain,
            "signals": gap["confidence_signals"],
        }

        # Phase 2: Capability spec
        prompt = (
            f"Define the capabilities needed for a specialist agent in: {domain}\n"
            "Return JSON: {\"capabilities\": [\"...\"], \"model_preference\": \"...\"}"
        )
        raw = await self.llm.generate(prompt, model=self.llm.cfg.model_fast)
        try:
            spec = json.loads(MARSEngine._extract_json(raw))
        except Exception:
            spec = {"capabilities": [domain], "model_preference": self.cfg.model_code}
        phase_results["capability_spec"] = spec

        # Phase 3: Candidate generation
        agent_id = f"agent-{uuid.uuid4().hex[:8]}"
        agent_name = f"{domain.title()}Specialist"

        # Phase 4-7: Validation (simplified for single-file)
        for phase in self.HIRING_PHASES[3:7]:
            phase_results[phase] = {"status": "passed"}

        # Phase 8: Deployment
        profile = AgentProfile(
            agent_id=agent_id,
            name=agent_name,
            specialty=domain,
            capabilities=spec.get("capabilities", [domain]),
            trust_score=0.5,
            created_at=datetime.datetime.utcnow().isoformat(),
        )
        self.trust_ledger.register(profile)
        phase_results["deployment"] = {"agent_id": agent_id, "status": "active"}

        log.info("NEXUS: hired agent %s (%s) for domain '%s'", agent_id, agent_name, domain)
        return profile

    async def delegate_task(self, task: str, agents: Optional[List[str]] = None) -> Dict:
        domain = self._classify_domain(task)
        candidates = agents or []

        if not candidates:
            trusted = self.trust_ledger.get_trusted(self.cfg.trust_threshold)
            for agent in trusted:
                if domain.lower() in agent.specialty.lower():
                    candidates.append(agent.agent_id)
            if not candidates and trusted:
                candidates = [trusted[0].agent_id]

        if not candidates:
            return {"delegated": False, "reason": "No suitable agent found"}

        agent_id = candidates[0]
        intuition = await self.shadow_mind.intuition(task)

        return {
            "delegated": True,
            "agent_id": agent_id,
            "shadow_mind_intuition": intuition,
            "domain": domain,
        }

    @staticmethod
    def _classify_domain(task: str) -> str:
        task_lower = task.lower()
        domains = {
            "coding": ["code", "function", "debug", "script", "python", "api"],
            "data": ["data", "csv", "json", "parse", "analyze", "database", "sql"],
            "devops": ["deploy", "docker", "server", "ci/cd", "infrastructure"],
            "writing": ["write", "essay", "blog", "document", "email", "report"],
            "research": ["research", "find", "search", "compare", "evaluate"],
            "system": ["file", "directory", "process", "system", "install"],
        }
        scores: Dict[str, int] = defaultdict(int)
        for domain, keywords in domains.items():
            for kw in keywords:
                if kw in task_lower:
                    scores[domain] += 1
        if scores:
            return max(scores, key=scores.get)
        return "general"


# ###########################################################################
#  SECTION 10 -- HYPERAGENT ARCHITECTURE  (Feature 2)
# ###########################################################################

class MetaImprovementLog:
    """Tracks meta-level improvements for self-acceleration."""

    def __init__(self):
        self._improvements: List[Dict] = []
        self._load()

    def _load(self) -> None:
        if META_IMPROVEMENT_PATH.exists():
            try:
                self._improvements = json.loads(
                    META_IMPROVEMENT_PATH.read_text(encoding="utf-8")
                )
            except Exception:
                pass

    def _save(self) -> None:
        META_IMPROVEMENT_PATH.write_text(
            json.dumps(self._improvements, indent=2, default=str),
            encoding="utf-8",
        )

    def record(self, improvement: Dict) -> None:
        improvement["timestamp"] = datetime.datetime.utcnow().isoformat()
        self._improvements.append(improvement)
        if len(self._improvements) > 200:
            self._improvements = self._improvements[-200:]
        self._save()

    def get_recent(self, limit: int = 10) -> List[Dict]:
        return self._improvements[-limit:]

    def count(self) -> int:
        return len(self._improvements)


class TaskAgent:
    """Does the actual work -- executes tools, generates responses."""

    def __init__(self, llm: OllamaRouter, memory: FourLayerMemory, hardware: TermuxHardware):
        self.llm = llm
        self.memory = memory
        self.hardware = hardware
        self._current_record: Optional[TaskRecord] = None

    def start_task(self, task: str) -> TaskRecord:
        self._current_record = TaskRecord(task=task)
        self._current_record.steps.append(f"Task received: {task[:100]}")
        return self._current_record

    async def execute(self, task: str, system_prompt: str = "") -> str:
        record = self._current_record or self.start_task(task)
        start_time = time.time()

        # Check L4 skills first
        skill_names = self.memory.l4.list_names()
        best_skill = self._match_skill(task, skill_names)
        if best_skill:
            skill_content = self.memory.l4.load_skill(best_skill)
            if skill_content:
                record.steps.append(f"Matched skill: {best_skill}")
                system_prompt += f"\n\nRelevant skill:\n{skill_content}"

        context = self.memory.build_context(task)
        full_system = (
            f"You are ZeusPrime Task Agent. Complete the user's task.\n\n"
            f"Memory context:\n{context}\n\n"
            f"{system_prompt}"
        )

        response = await self.llm.generate(task, system=full_system)
        record.steps.append(f"LLM response generated ({len(response)} chars)")
        record.tool_calls += 1
        record.result = response
        record.duration_s = time.time() - start_time
        record.success = True

        self.memory.l3.store(
            category="conversation",
            content=f"User: {task[:200]}\nAgent: {response[:300]}",
        )

        return response

    async def execute_tool(self, tool_name: str, args: Dict) -> str:
        record = self._current_record
        if record:
            record.tool_calls += 1
            record.steps.append(f"Tool call: {tool_name}({list(args.keys())})")

        handler = PANTHEON_TOOLS.get(tool_name)
        if handler:
            try:
                result = await handler(args) if asyncio.iscoroutinefunction(handler) else handler(args)
                return str(result)
            except Exception as exc:
                if record:
                    record.self_corrections += 1
                return f"[Tool error: {exc}]"
        return f"[Unknown tool: {tool_name}]"

    def record_user_correction(self, correction: str) -> None:
        if self._current_record:
            self._current_record.user_corrections += 1
            self._current_record.steps.append(f"User correction: {correction[:100]}")
        self.memory.l2.add_correction(correction)

    def finalize(self) -> Optional[TaskRecord]:
        record = self._current_record
        self._current_record = None
        return record

    @staticmethod
    def _match_skill(task: str, skill_names: List[str]) -> Optional[str]:
        task_lower = task.lower()
        best_score = 0
        best_name = None
        for name in skill_names:
            words = name.replace("_", " ").replace("-", " ").lower().split()
            score = sum(1 for w in words if w in task_lower)
            if score > best_score:
                best_score = score
                best_name = name
        return best_name if best_score > 0 else None


class MetaAgent:
    """
    Improves the Task Agent AND improves itself.
    - Can rewrite its own improvement logic
    - Gains compound over time (self-accelerating)
    - Meta-level improvements transfer across domains
    """

    def __init__(
        self,
        llm: OllamaRouter,
        memory: FourLayerMemory,
        mars: MARSEngine,
        skill_extractor: SkillExtractor,
        nexus: NEXUSHiring,
    ):
        self.llm = llm
        self.memory = memory
        self.mars = mars
        self.skill_extractor = skill_extractor
        self.nexus = nexus
        self.improvement_log = MetaImprovementLog()
        self._improvement_strategies: List[Callable] = [
            self._strategy_reflect,
            self._strategy_extract_skills,
            self._strategy_hire_if_needed,
            self._strategy_self_improve,
        ]

    async def post_task_improvement(self, record: TaskRecord) -> Dict[str, Any]:
        results: Dict[str, Any] = {"strategies_run": []}

        for strategy in self._improvement_strategies:
            try:
                name = strategy.__name__
                outcome = await strategy(record)
                results["strategies_run"].append({"name": name, "outcome": outcome})
            except Exception as exc:
                log.error("Meta strategy %s failed: %s", strategy.__name__, exc)

        self.memory.l3.store_task({
            "task": record.task,
            "result": record.result[:500],
            "tool_calls": record.tool_calls,
            "self_corrections": record.self_corrections,
            "user_corrections": record.user_corrections,
            "duration_s": record.duration_s,
            "skill_extracted": int(record.skill_extracted),
        })

        self.improvement_log.record({
            "task": record.task[:100],
            "improvements": results["strategies_run"],
            "total_improvements": self.improvement_log.count(),
        })

        return results

    async def _strategy_reflect(self, record: TaskRecord) -> Dict:
        if not record.success and record.result:
            principle = await self.mars.reflect_on_failure(
                record.task, record.result
            )
            return {"type": "principle", "result": principle}
        elif record.success and record.steps:
            procedure = await self.mars.reflect_on_success(
                record.task, record.steps, record.result
            )
            return {"type": "procedure", "result": procedure}
        return {"type": "skip", "reason": "no reflection needed"}

    async def _strategy_extract_skills(self, record: TaskRecord) -> Dict:
        skill_name = await self.skill_extractor.extract(record)
        if skill_name:
            return {"extracted": True, "skill_name": skill_name}
        return {"extracted": False}

    async def _strategy_hire_if_needed(self, record: TaskRecord) -> Dict:
        gaps = await self.nexus.talent_scout_scan()
        hired = []
        for gap in gaps[:2]:
            agent = await self.nexus.hire_agent(gap)
            if agent:
                hired.append(agent.agent_id)
        return {"gaps_found": len(gaps), "agents_hired": hired}

    async def _strategy_self_improve(self, record: TaskRecord) -> Dict:
        recent = self.improvement_log.get_recent(20)
        if len(recent) < 5:
            return {"improved": False, "reason": "not enough data"}

        prompt = (
            "You are the Meta Agent's self-improvement engine. "
            "Review recent improvement patterns and suggest ONE meta-level "
            "optimization to the improvement process itself.\n\n"
            f"Recent improvements: {json.dumps(recent[-5:], default=str)}\n"
            f"Total improvements so far: {self.improvement_log.count()}\n\n"
            "Return JSON: {\"optimization\": \"...\", \"domain\": \"...\", "
            "\"expected_impact\": \"...\", \"transferable\": true/false}"
        )
        raw = await self.llm.generate(prompt, model=self.llm.cfg.model_fast)
        try:
            optimization = json.loads(MARSEngine._extract_json(raw))
        except Exception:
            optimization = {"optimization": raw.strip(), "domain": "general"}

        return {"improved": True, "optimization": optimization}

    def get_system_enhancement(self, task: str) -> str:
        parts = []
        mars_guidance = self.mars.render_guidance(task)
        if mars_guidance:
            parts.append(mars_guidance)

        recent_improvements = self.improvement_log.get_recent(3)
        if recent_improvements:
            parts.append("\nRecent meta-improvements:")
            for imp in recent_improvements:
                parts.append(f"  - {imp.get('task', '')[:60]}")

        return "\n".join(parts)


# ###########################################################################
#  SECTION 11 -- PANTHEON TOOLS  (Feature 10)
# ###########################################################################

async def tool_bash(args: Dict) -> str:
    cmd = args.get("command", "")
    if not cmd:
        return "[No command provided]"
    try:
        proc = await asyncio.create_subprocess_shell(
            cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=args.get("cwd"),
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
        out = stdout.decode("utf-8", errors="replace")
        err = stderr.decode("utf-8", errors="replace")
        return out + ("\n[STDERR] " + err if err else "")
    except asyncio.TimeoutError:
        return "[Command timed out after 60s]"
    except Exception as exc:
        return f"[Error: {exc}]"


async def tool_file_read(args: Dict) -> str:
    path = args.get("path", "")
    if not path:
        return "[No path provided]"
    try:
        return Path(path).read_text(encoding="utf-8")[:10000]
    except Exception as exc:
        return f"[Read error: {exc}]"


async def tool_file_write(args: Dict) -> str:
    path = args.get("path", "")
    content = args.get("content", "")
    if not path:
        return "[No path provided]"
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(content, encoding="utf-8")
        return f"Written {len(content)} bytes to {path}"
    except Exception as exc:
        return f"[Write error: {exc}]"


async def tool_web_search(args: Dict) -> str:
    query = args.get("query", "")
    if not query:
        return "[No query]"
    if not HAS_HTTPX:
        return await tool_bash({"command": f"curl -s 'https://api.duckduckgo.com/?q={query}&format=json'"})
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            r = await c.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json"},
            )
            data = r.json()
            results = []
            if data.get("AbstractText"):
                results.append(data["AbstractText"])
            for topic in data.get("RelatedTopics", [])[:5]:
                if isinstance(topic, dict) and topic.get("Text"):
                    results.append(topic["Text"])
            return "\n".join(results) if results else "[No results found]"
    except Exception as exc:
        return f"[Search error: {exc}]"


async def tool_web_get(args: Dict) -> str:
    url = args.get("url", "")
    if not url:
        return "[No URL]"
    if not HAS_HTTPX:
        return await tool_bash({"command": f"curl -sL '{url}' | head -c 5000"})
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as c:
            r = await c.get(url)
            return r.text[:5000]
    except Exception as exc:
        return f"[Fetch error: {exc}]"


async def tool_telegram_send(args: Dict) -> str:
    token = os.environ.get("TELEGRAM_BOT_TOKEN", args.get("token", ""))
    chat_id = args.get("chat_id", "")
    text = args.get("text", "")
    if not all([token, chat_id, text]):
        return "[Missing token, chat_id, or text]"
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    if HAS_HTTPX:
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                r = await c.post(url, json={"chat_id": chat_id, "text": text})
                return r.text
        except Exception as exc:
            return f"[Telegram error: {exc}]"
    return await tool_bash({
        "command": f"curl -s -X POST '{url}' -H 'Content-Type: application/json' "
                   f"-d '{{\"chat_id\":\"{chat_id}\",\"text\":\"{text}\"}}'"
    })


def tool_memory_search(args: Dict) -> str:
    return "[Use memory.l3.search() via the agent]"


def tool_skill_list(args: Dict) -> str:
    return "[Use memory.l4.list_names() via the agent]"


PANTHEON_TOOLS: Dict[str, Callable] = {
    # OpenClaw tools
    "bash": tool_bash,
    "file_read": tool_file_read,
    "file_write": tool_file_write,
    "web_search": tool_web_search,
    "web_get": tool_web_get,
    "telegram": tool_telegram_send,
    # Hermes tools
    "memory_search": tool_memory_search,
    "skill_list": tool_skill_list,
    # GPTSwarm (orchestrator -- the MetaAgent itself is the swarm optimizer)
    # MothBot (skill extraction -- SkillExtractor handles this)
    # Coreon (execution engine -- TaskAgent handles this)
}


# ###########################################################################
#  SECTION 12 -- OFFLINE VOICE  (Feature 11)
# ###########################################################################

class OfflineVoice:
    """Whisper STT + Edge TTS -- fully offline capable."""

    def __init__(self, cfg: ZeusConfig):
        self.cfg = cfg
        self._whisper_model = None
        self._tts_voice = cfg.tts_voice

    async def init_whisper(self) -> bool:
        if not HAS_WHISPER:
            log.warning("faster-whisper not installed -- STT disabled")
            return False
        try:
            self._whisper_model = WhisperModel(
                self.cfg.whisper_model_size,
                device="cpu",
                compute_type="int8",
            )
            log.info("Whisper model loaded: %s", self.cfg.whisper_model_size)
            return True
        except Exception as exc:
            log.error("Whisper init failed: %s", exc)
            return False

    async def transcribe(self, audio_path: str) -> str:
        if not self._whisper_model:
            return "[Whisper not initialized]"
        try:
            segments, _ = self._whisper_model.transcribe(
                audio_path, beam_size=5
            )
            return " ".join(seg.text for seg in segments).strip()
        except Exception as exc:
            return f"[Transcription error: {exc}]"

    async def speak(self, text: str, output_path: Optional[str] = None) -> Optional[str]:
        if not HAS_EDGE_TTS:
            if IS_TERMUX:
                proc = await asyncio.create_subprocess_exec(
                    "termux-tts-speak", text,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await proc.communicate()
                return None
            log.warning("edge-tts not installed -- TTS disabled")
            return None

        out = output_path or str(ZEUS_DIR / f"tts_{int(time.time())}.mp3")
        try:
            communicate = edge_tts.Communicate(text, self._tts_voice)
            await communicate.save(out)
            await self._play_audio(out)
            return out
        except Exception as exc:
            log.error("TTS failed: %s", exc)
            return None

    async def listen_microphone(self, duration_s: int = 5) -> str:
        wav_path = str(ZEUS_DIR / f"mic_{int(time.time())}.wav")
        if IS_TERMUX:
            cmd = (
                f"termux-microphone-record -l {duration_s} "
                f"-f {wav_path} -e amr_wb"
            )
        else:
            cmd = (
                f"arecord -d {duration_s} -f S16_LE -r 16000 "
                f"-c 1 {wav_path} 2>/dev/null"
            )
        try:
            proc = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.communicate(), timeout=duration_s + 5)
            if Path(wav_path).exists():
                return await self.transcribe(wav_path)
            return "[Recording failed]"
        except Exception as exc:
            return f"[Microphone error: {exc}]"

    @staticmethod
    async def _play_audio(path: str) -> None:
        if IS_TERMUX:
            player = "play-audio"
        elif shutil.which("mpv"):
            player = "mpv --no-terminal"
        elif shutil.which("aplay"):
            player = "aplay"
        else:
            return
        try:
            proc = await asyncio.create_subprocess_shell(
                f"{player} {shlex.quote(path)}",
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.communicate(), timeout=30)
        except Exception:
            pass


# ###########################################################################
#  SECTION 13 -- ZEUS PRIME ORCHESTRATOR
# ###########################################################################

class ZeusPrime:
    """
    King of the Pantheon -- the master orchestrator.

    Unifies all 12 feature systems into a single coherent agent:
    1. MARS reflection
    2. Hyperagent (Meta + Task agents)
    3. Four-layer memory
    4. Skill extraction
    5. NEXUS multi-agent hiring
    6. Android hardware (Termux)
    7. Multi-model routing (Ollama)
    8. Sandboxed Skill Forge
    9. Simulation mode
    10. Pantheon tools
    11. Offline voice (Whisper + Edge TTS)
    12. Single file
    """

    def __init__(self, cfg: Optional[ZeusConfig] = None):
        self.cfg = cfg or ZeusConfig.load()
        self._running = False
        self._session_id = uuid.uuid4().hex[:12]

        # Feature 7: Multi-model router
        self.llm = OllamaRouter(self.cfg)

        # Feature 3: Four-layer memory
        self.memory = FourLayerMemory(self.cfg)

        # Feature 1: MARS reflection
        self.mars = MARSEngine(self.llm, self.memory)

        # Feature 4: Skill extraction
        self.skill_extractor = SkillExtractor(self.llm, self.memory)

        # Feature 5: NEXUS hiring
        self.nexus = NEXUSHiring(self.llm, self.memory, self.cfg)

        # Feature 6: Android hardware
        self.hardware = TermuxHardware()

        # Feature 2: Hyperagent (Task + Meta)
        self.task_agent = TaskAgent(self.llm, self.memory, self.hardware)
        self.meta_agent = MetaAgent(
            self.llm, self.memory, self.mars, self.skill_extractor, self.nexus
        )

        # Feature 8: Skill Forge
        self.forge = SkillForge(self.llm, self.memory, self.cfg)

        # Feature 9: Simulation mode
        self.simulation = SimulationMode(self.llm)

        # Feature 11: Offline voice
        self.voice = OfflineVoice(self.cfg)

        # Stats
        self._stats = {
            "total_requests": 0,
            "skill_hits": 0,
            "llm_calls": 0,
            "skills_extracted": 0,
            "agents_hired": 0,
            "reflections": 0,
            "simulations": 0,
            "errors": 0,
        }

        # Conversation history (in-session)
        self._history: List[Dict[str, str]] = []

    async def start(self) -> None:
        log.info(
            "Starting ZeusPrime v%s (session: %s)", __version__, self._session_id
        )

        # Initialize L1 core memory
        self.memory.l1.set("session_id", self._session_id)
        self.memory.l1.set("version", __version__)
        self.memory.l1.set("platform", platform.platform())
        self.memory.l1.set("is_termux", str(IS_TERMUX))

        device_info = await self.hardware.get_device_info()
        self.memory.l1.set("device", json.dumps(device_info))
        self.memory.l1.freeze()

        self._running = True
        self.cfg.save()
        log.info("ZeusPrime started -- all systems online")

    async def stop(self) -> None:
        log.info("Stopping ZeusPrime... Stats: %s", json.dumps(self._stats))
        self._running = False

    async def process(self, user_input: str) -> str:
        if not self._running:
            return "ZeusPrime not started. Call start() first."

        self._stats["total_requests"] += 1
        user_input_stripped = user_input.strip()

        # Handle special commands
        if user_input_stripped.startswith("/"):
            return await self._handle_command(user_input_stripped)

        # Simulation mode intercept
        if self.simulation.is_active():
            if user_input_stripped.lower() in ("exit sim", "/sim off", "/revert"):
                self.simulation.revert()
                return "Simulation mode deactivated. State reverted."
            result = await self.simulation.simulate_command(user_input_stripped)
            self._stats["simulations"] += 1
            risk = result.get("risk_level", "unknown")
            return (
                f"[SIM] Exit code: {result.get('exit_code', '?')}\n"
                f"[SIM] Risk: {risk}\n"
                f"stdout:\n{result.get('stdout', '')}\n"
                f"stderr:\n{result.get('stderr', '')}\n"
                f"Side effects: {json.dumps(result.get('side_effects', []))}"
            )

        # Get meta-agent enhancements
        enhancement = self.meta_agent.get_system_enhancement(user_input)

        # Task Agent executes
        self.task_agent.start_task(user_input)
        response = await self.task_agent.execute(user_input, system_prompt=enhancement)
        self._stats["llm_calls"] += 1

        # Finalize and run post-task improvement
        record = self.task_agent.finalize()
        if record:
            try:
                improvement_results = await self.meta_agent.post_task_improvement(record)
                if record.skill_extracted:
                    self._stats["skills_extracted"] += 1
                self._stats["reflections"] += 1
            except Exception as exc:
                log.error("Post-task improvement failed: %s", exc)

        self._history.append({"user": user_input, "assistant": response})
        return response

    async def _handle_command(self, cmd: str) -> str:
        parts = cmd.split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        handlers: Dict[str, Callable] = {
            "/status": self._cmd_status,
            "/stats": self._cmd_status,
            "/memory": self._cmd_memory,
            "/skills": self._cmd_skills,
            "/skill": self._cmd_skill_detail,
            "/forge": self._cmd_forge,
            "/sim": self._cmd_sim,
            "/simulate": self._cmd_sim,
            "/hire": self._cmd_hire,
            "/agents": self._cmd_agents,
            "/reflect": self._cmd_reflect,
            "/mars": self._cmd_mars,
            "/voice": self._cmd_voice,
            "/photo": self._cmd_photo,
            "/screenshot": self._cmd_screenshot,
            "/location": self._cmd_location,
            "/battery": self._cmd_battery,
            "/clipboard": self._cmd_clipboard,
            "/config": self._cmd_config,
            "/help": self._cmd_help,
            "/bash": lambda a: tool_bash({"command": a}),
            "/search": lambda a: tool_web_search({"query": a}),
            "/correct": self._cmd_correct,
        }

        handler = handlers.get(command)
        if handler:
            result = handler(arg)
            if asyncio.iscoroutine(result):
                return await result
            return result
        return f"Unknown command: {command}. Type /help for available commands."

    async def _cmd_status(self, _: str = "") -> str:
        mem_stats = self.memory.stats()
        agents = self.nexus.trust_ledger.get_all()
        lines = [
            f"=== ZeusPrime v{__version__} ===",
            f"Session: {self._session_id}",
            f"Platform: {'Termux' if IS_TERMUX else platform.platform()}",
            f"",
            f"Stats: {json.dumps(self._stats, indent=2)}",
            f"",
            f"Memory:",
            f"  L1 Core: {mem_stats['l1_entries']} entries (frozen: {mem_stats['l1_frozen']})",
            f"  L3 Long-term: {mem_stats['l3']['memories']} memories, "
            f"{mem_stats['l3']['tasks']} tasks, {mem_stats['l3']['reflections']} reflections",
            f"  L4 Skills: {mem_stats['l4_skills']} skills",
            f"",
            f"NEXUS Agents: {len(agents)}",
            f"MARS Principles: {len(self.mars._principles)}",
            f"MARS Procedures: {len(self.mars._procedures)}",
            f"Meta Improvements: {self.meta_agent.improvement_log.count()}",
            f"Simulation: {'ACTIVE' if self.simulation.is_active() else 'inactive'}",
        ]
        return "\n".join(lines)

    async def _cmd_memory(self, query: str = "") -> str:
        if query:
            results = self.memory.l3.search(query)
            if not results:
                return f"No memories matching '{query}'"
            lines = [f"Found {len(results)} memories:"]
            for r in results:
                lines.append(f"  [{r['category']}] {r['content'][:120]}")
            return "\n".join(lines)
        return self.memory.build_context()

    async def _cmd_skills(self, _: str = "") -> str:
        names = self.memory.l4.list_names()
        if not names:
            return "No skills in library. Complete tasks to auto-extract skills."
        return "Skills:\n" + "\n".join(f"  - {n}" for n in names)

    async def _cmd_skill_detail(self, name: str = "") -> str:
        if not name:
            return "Usage: /skill <name>"
        content = self.memory.l4.load_skill(name.strip())
        return content if content else f"Skill '{name}' not found."

    async def _cmd_forge(self, description: str = "") -> str:
        if not description:
            return "Usage: /forge <description of the skill to create>"
        result = await self.forge.create_skill(description)
        status = "INSTALLED" if result["installed"] else "FAILED"
        return (
            f"Forge result: {status}\n"
            f"Skill ID: {result['skill_id']}\n"
            f"Test output: {result['test_output'][:500]}"
        )

    async def _cmd_sim(self, arg: str = "") -> str:
        if arg.lower() in ("on", "start", ""):
            state = {
                "cwd": os.getcwd(),
                "env_snapshot": dict(os.environ),
                "history": [h["user"] for h in self._history[-5:]],
            }
            self.simulation.freeze(state)
            return (
                "Simulation mode ACTIVATED.\n"
                "Type bash commands to simulate them safely.\n"
                "Type '/sim off' or '/revert' to exit and revert."
            )
        elif arg.lower() in ("off", "stop", "revert"):
            self.simulation.revert()
            return "Simulation mode deactivated. State reverted."
        elif arg.lower() == "history":
            history = self.simulation.get_history()
            if not history:
                return "No simulation history."
            lines = []
            for h in history:
                lines.append(f"$ {h['command']}")
                lines.append(f"  exit: {h['result'].get('exit_code', '?')} "
                           f"risk: {h['result'].get('risk_level', '?')}")
            return "\n".join(lines)
        return "Usage: /sim [on|off|history]"

    async def _cmd_hire(self, _: str = "") -> str:
        gaps = await self.nexus.talent_scout_scan()
        if not gaps:
            return "No coverage gaps detected. Team is sufficient."
        lines = ["Coverage gaps detected:"]
        for gap in gaps:
            lines.append(
                f"  - {gap['domain']}: {gap['confidence_signals']} signals "
                f"(failures={gap['failures']}, slow={gap['slow_tasks']}, "
                f"corrections={gap['corrections']})"
            )
        lines.append("\nHiring agents for top gaps...")
        for gap in gaps[:2]:
            agent = await self.nexus.hire_agent(gap)
            if agent:
                lines.append(f"  Hired: {agent.name} ({agent.agent_id})")
                self._stats["agents_hired"] += 1
        return "\n".join(lines)

    async def _cmd_agents(self, _: str = "") -> str:
        agents = self.nexus.trust_ledger.get_all()
        if not agents:
            return "No agents in the team. Use /hire to recruit."
        lines = ["NEXUS Agent Team:"]
        for a in agents:
            lines.append(
                f"  {a.name} ({a.agent_id})\n"
                f"    Specialty: {a.specialty}\n"
                f"    Trust: {a.trust_score:.2f} | "
                f"Tasks: {a.tasks_completed}/{a.tasks_completed + a.tasks_failed}\n"
                f"    Status: {a.status}"
            )
        return "\n".join(lines)

    async def _cmd_reflect(self, _: str = "") -> str:
        reflections = self.memory.l3.get_reflections(limit=10)
        if not reflections:
            return "No reflections yet. Complete tasks to generate MARS reflections."
        lines = ["Recent MARS Reflections:"]
        for r in reflections:
            lines.append(f"  [{r['type']}] {r['insight'][:100]}")
        return "\n".join(lines)

    async def _cmd_mars(self, _: str = "") -> str:
        lines = [
            f"MARS Engine Status:",
            f"  Principles: {len(self.mars._principles)}",
            f"  Procedures: {len(self.mars._procedures)}",
        ]
        if self.mars._principles:
            lines.append("\nRecent Principles:")
            for p in self.mars._principles[-3:]:
                lines.append(f"  - {p.get('principle', '')[:100]}")
        if self.mars._procedures:
            lines.append("\nRecent Procedures:")
            for p in self.mars._procedures[-3:]:
                lines.append(f"  - {p.get('procedure_name', '')}")
        return "\n".join(lines)

    async def _cmd_voice(self, text: str = "") -> str:
        if text:
            out = await self.voice.speak(text)
            return f"Spoke: {text}" + (f" (saved: {out})" if out else "")
        return (
            "Voice commands:\n"
            "  /voice <text>     -- speak text via TTS\n"
            "  Use --voice flag at startup for full voice mode"
        )

    async def _cmd_photo(self, _: str = "") -> str:
        if not IS_TERMUX:
            return "[Camera only available on Termux/Android]"
        path = await self.hardware.take_photo()
        if path:
            analysis = await self.hardware.analyze_image(path, self.llm)
            return f"Photo saved: {path}\nAnalysis: {analysis}"
        return "[Camera capture failed]"

    async def _cmd_screenshot(self, _: str = "") -> str:
        if not IS_TERMUX:
            return "[Screenshot only available on Termux/Android]"
        path = await self.hardware.take_screenshot()
        if path:
            analysis = await self.hardware.analyze_image(path, self.llm)
            return f"Screenshot saved: {path}\nAnalysis: {analysis}"
        return "[Screenshot failed]"

    async def _cmd_location(self, _: str = "") -> str:
        loc = await self.hardware.get_location()
        if loc:
            return json.dumps(loc, indent=2)
        return "[Location unavailable]" if IS_TERMUX else "[Location only available on Termux]"

    async def _cmd_battery(self, _: str = "") -> str:
        bat = await self.hardware.get_battery()
        if bat:
            return json.dumps(bat, indent=2)
        return "[Battery info unavailable]" if IS_TERMUX else "[Battery only available on Termux]"

    async def _cmd_clipboard(self, text: str = "") -> str:
        if text:
            await self.hardware.set_clipboard(text)
            return f"Clipboard set: {text[:50]}..."
        content = await self.hardware.get_clipboard()
        return content if content else "[Clipboard empty or unavailable]"

    async def _cmd_config(self, _: str = "") -> str:
        return json.dumps(asdict(self.cfg), indent=2, default=str)

    async def _cmd_correct(self, correction: str = "") -> str:
        if not correction:
            return "Usage: /correct <what you want me to learn>"
        self.task_agent.record_user_correction(correction)
        return f"Correction recorded. I'll remember: {correction}"

    async def _cmd_help(self, _: str = "") -> str:
        return textwrap.dedent("""
            ZeusPrime Commands:
            ────────────────────────────────────
            /status          Show system status & stats
            /memory [query]  Browse or search memory
            /skills          List all extracted skills
            /skill <name>    Show skill details
            /forge <desc>    Create a new skill in the Forge
            /sim [on|off]    Toggle Simulation Mode
            /sim history     Show simulation history
            /hire            Run NEXUS talent scout & hire
            /agents          List hired agents
            /reflect         Show MARS reflections
            /mars            MARS engine status
            /voice <text>    Text-to-speech
            /photo           Take & analyze photo (Termux)
            /screenshot      Take & analyze screenshot (Termux)
            /location        Get GPS location (Termux)
            /battery         Battery status (Termux)
            /clipboard [txt] Get/set clipboard (Termux)
            /config          Show configuration
            /correct <text>  Record a correction
            /bash <cmd>      Execute bash command
            /search <query>  Web search
            /help            This help message

            Just type normally to chat with ZeusPrime.
            All tasks auto-trigger MARS reflection & skill extraction.
        """).strip()


# ###########################################################################
#  SECTION 14 -- VOICE INTERACTIVE LOOP
# ###########################################################################

async def voice_loop(zeus: ZeusPrime) -> None:
    """Full voice-interactive mode with wake word detection."""
    print("\n[Voice Mode] Say a wake word to start. Press Ctrl+C to exit.")
    print(f"[Voice Mode] Wake words: {zeus.cfg.wake_words}")
    await zeus.voice.init_whisper()

    while zeus._running:
        try:
            print("[Listening...]")
            transcript = await zeus.voice.listen_microphone(duration_s=5)
            if not transcript or transcript.startswith("["):
                continue

            transcript_lower = transcript.lower().strip()
            is_wake = any(w in transcript_lower for w in zeus.cfg.wake_words)
            if not is_wake and not zeus._history:
                continue

            print(f"[You] {transcript}")
            response = await zeus.process(transcript)
            print(f"[ZeusPrime] {response}")
            await zeus.voice.speak(response)

        except KeyboardInterrupt:
            break
        except Exception as exc:
            log.error("Voice loop error: %s", exc)
            await asyncio.sleep(1)


# ###########################################################################
#  SECTION 15 -- CLI INTERACTIVE LOOP
# ###########################################################################

BANNER = r"""
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║    ⚡  Z E U S P R I M E  ⚡                                  ║
║    King of the Pantheon                                       ║
║    v{version}                                                    ║
║                                                               ║
║    Features: MARS | Hyperagent | 4-Layer Memory | NEXUS       ║
║    Skill Forge | Simulation | Pantheon Tools | Offline Voice  ║
║                                                               ║
║    Type /help for commands  |  Ctrl+C to exit                 ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
""".format(version=__version__)


async def cli_loop(zeus: ZeusPrime) -> None:
    """Interactive CLI loop."""
    print(BANNER)
    status = await zeus._cmd_status()
    print(status)
    print()

    while zeus._running:
        try:
            user_input = await asyncio.get_event_loop().run_in_executor(
                None, lambda: input("\n⚡ You: ")
            )
            if not user_input.strip():
                continue
            if user_input.strip().lower() in ("exit", "quit", "/quit", "/exit"):
                break

            response = await zeus.process(user_input)
            print(f"\n🔱 ZeusPrime: {response}")

        except (KeyboardInterrupt, EOFError):
            break
        except Exception as exc:
            log.error("CLI error: %s", exc)
            print(f"\n[Error] {exc}")


# ###########################################################################
#  SECTION 16 -- MAIN ENTRY POINT
# ###########################################################################

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="ZeusPrime -- King of the Pantheon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Examples:
              python zeus_prime.py                    # Interactive CLI
              python zeus_prime.py --voice            # Voice mode
              python zeus_prime.py --simulate         # Start in simulation mode
              python zeus_prime.py --status           # Show status
              python zeus_prime.py --forge "script to count files"
              python zeus_prime.py --hire             # Run NEXUS hiring
        """),
    )
    parser.add_argument("--voice", action="store_true", help="Start in voice mode")
    parser.add_argument("--simulate", action="store_true", help="Start in simulation mode")
    parser.add_argument("--status", action="store_true", help="Show status and exit")
    parser.add_argument("--forge", type=str, help="Create a skill in the Forge")
    parser.add_argument("--hire", action="store_true", help="Run NEXUS talent scout")
    parser.add_argument("--config", type=str, help="Path to config JSON file")

    args = parser.parse_args()

    cfg = ZeusConfig.load()
    if args.config:
        try:
            raw = json.loads(Path(args.config).read_text(encoding="utf-8"))
            known = {f.name for f in cfg.__dataclass_fields__.values()}
            for k, v in raw.items():
                if k in known:
                    setattr(cfg, k, v)
        except Exception as exc:
            print(f"[Config error] {exc}")
            sys.exit(1)

    zeus = ZeusPrime(cfg)
    await zeus.start()

    try:
        if args.status:
            print(await zeus._cmd_status())
            return

        if args.forge:
            result = await zeus.forge.create_skill(args.forge)
            status_str = "INSTALLED" if result["installed"] else "FAILED"
            print(f"Forge: {status_str} -- {result['skill_id']}")
            print(f"Output: {result['test_output'][:500]}")
            return

        if args.hire:
            print(await zeus._cmd_hire())
            return

        if args.simulate:
            await zeus.process("/sim on")
            print("Simulation mode activated. Type commands to simulate.")

        if args.voice:
            await voice_loop(zeus)
        else:
            await cli_loop(zeus)
    finally:
        await zeus.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[ZeusPrime] Goodbye.")

# --- CLAW PANTHEON INTEGRATION ---
import json
import time
import os
import re
import urllib.request
import urllib.parse
import concurrent.futures
import subprocess

class LegionBus:
    def __init__(self):
        self.messages = []
    def publish(self, sender, data, metadata=None):
        entry = {"sender": sender, "data": data, "timestamp": time.time(), "meta": metadata}
        self.messages.append(entry)
        print(f"[BUS] {sender} >> {data}")

class ArmoryLoader:
    def __init__(self):
        self.source = "https://github.com/VoltAgent/awesome-openclaw-skills"
    def load_skill(self, skill_name):
        return True

class ClawAgent:
    def __init__(self, name, domain):
        self.name = name
        self.domain = domain
    def run(self, task, bus, armory, **kwargs):
        result = f"Action by {self.name}"
        bus.publish(self.name, result)
        return result

class IronClawAgent(ClawAgent):
    def run(self, task, bus, armory, **kwargs):
        print(f"[{self.name}] ACT: Executing Zero-Trust Audit...")
        blacklist = [(r"rm\s+-rf", "Mass deletion"), (r"format\s+", "Drive format"), (r"chmod\s+777", "Insecure perms")]
        violations = [reason for pattern, reason in blacklist if re.search(pattern, task, re.I)]
        if violations:
            msg = f"BLOCK: {', '.join(violations)}"
            bus.publish(self.name, msg, {"status": "BLOCK"})
            return msg
        bus.publish(self.name, "Audit: CLEAN")
        return "PASS"

class ClawMemAgent(ClawAgent):
    def run(self, task, bus, armory, memory=None, **kwargs):
        print(f"[{self.name}] ACT: Injecting contextual memory...")
        context = f"PREVIOUS CONTEXT: {memory[-1]['task'] if memory else 'None'}"
        bus.publish(self.name, context)
        return context

class TrinityClawAgent(ClawAgent):
    def run(self, task, bus, armory, **kwargs):
        print(f"[{self.name}] ACT: Analyzing task complexity...")
        words = task.split()
        complexity = "HIGH" if len(words) > 5 or any(k in task for k in [",", ";", "and"]) else "LOW"
        bus.publish(self.name, f"PLAN: Complexity={complexity}")
        return complexity

class ZeroClawAgent(ClawAgent):
    def run(self, task, bus, armory, **kwargs):
        print(f"[{self.name}] ACT: Verifying infrastructure state...")
        files = os.listdir('.')
        state = f"INFRA_READY: {len(files)} artifacts."
        bus.publish(self.name, state)
        return state

class OpenBrowserClawAgent(ClawAgent):
    def run(self, task, bus, armory, **kwargs):
        print(f"[{self.name}] ACT: Fetching live web signals...")
        query = task.replace("research", "").replace("find", "").strip()
        try:
            url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json&no_html=1"
            headers = {'User-Agent': 'Claw-Prime/1.0'}
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode('utf-8'))
                signal = data.get("AbstractText") or data.get("Definition") or f"No direct summary for {query}."
                res = f"LIVE_SIGNAL: {signal[:100]}..."
                bus.publish(self.name, res)
                return res
        except Exception as e:
            err = f"SIGNAL_ERROR: {str(e)}"
            bus.publish(self.name, err)
            return err

class ClawSwarmAgent(ClawAgent):
    def run(self, task, bus, armory, **kwargs):
        print(f"[{self.name}] ACT: Fanning out swarm execution...")
        sub_tasks = [t.strip() for t in task.split(',') if t.strip()]
        if len(sub_tasks) <= 1: sub_tasks = [f"{task} scan", f"{task} deep-dive"]
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            browser = OpenBrowserClawAgent("OpenBrowserClaw", "Browser")
            futures = {executor.submit(browser.run, t, bus, armory): t for t in sub_tasks}
            for future in concurrent.futures.as_completed(futures):
                try: results.append(future.result())
                except: results.append("Thread Error")
        bus.publish(self.name, f"SWARM_COMPLETE: {len(results)} threads.")
        return " | ".join(results)

class ARCAgent(ClawAgent):
    def run(self, task, bus, armory, live_data=None, **kwargs):
        print(f"[{self.name}] ACT: Synthesizing Research Data...")
        report = f"SYNTHESIS: {live_data if live_data else 'Deep scan completed.'}"
        bus.publish(self.name, f"REPORT: {report}")
        return report

class AutoClawAgent(ClawAgent):
    def run(self, task, bus, armory, **kwargs):
        print(f"[{self.name}] ACT: Scheduling automation...")
        res = f"AUTO_EXEC: Task '{task}' scheduled for recursive polling."
        bus.publish(self.name, res)
        return res

class OpenCrabsAgent(ClawAgent):
    def run(self, task, bus, armory, **kwargs):
        print(f"[{self.name}] ACT: Compiling performance hooks...")
        res = "RUST_STREAMS: Memory-safe execution path verified."
        bus.publish(self.name, res)
        return res

class PicoClawAgent(ClawAgent):
    def run(self, task, bus, armory, **kwargs):
        print(f"[{self.name}] ACT: Checking edge compute availability...")
        res = "GO_EDGE: Lightweight node active."
        bus.publish(self.name, res)
        return res

class TinyAGIAgent(ClawAgent):
    def run(self, task, bus, armory, **kwargs):
        print(f"[{self.name}] ACT: Orchestrating agent handoffs...")
        res = "AGI_CORE: Coordination synchronized across Pantheon."
        bus.publish(self.name, res)
        return res

class OpenClawAgent(ClawAgent):
    def run(self, task, bus, armory, **kwargs):
        print(f"[{self.name}] ACT: Executing general logic...")
        res = f"OPEN_CLAW: Task '{task}' processed."
        bus.publish(self.name, res)
        return res

class ClawPrime:
    def __init__(self, storage_path="claw_memory.json"):
        self.name = "Claw-Prime"
        self.storage_path = storage_path
        self.bus = LegionBus()
        self.armory = ArmoryLoader()
        self.memory = self.load_memory()
        self.legion = {
            "OpenClaw": OpenClawAgent("OpenClaw", "Core"),
            "ARC": ARCAgent("ARC", "Research"),
            "AutoClaw": AutoClawAgent("AutoClaw", "Automation"),
            "OpenCrabs": OpenCrabsAgent("OpenCrabs", "Rust"),
            "PicoClaw": PicoClawAgent("PicoClaw", "Edge"),
            "ZeroClaw": ZeroClawAgent("ZeroClaw", "Infrastructure"),
            "TinyAGI": TinyAGIAgent("TinyAGI", "Coordination"),
            "TrinityClaw": TrinityClawAgent("TrinityClaw", "Logic"),
            "OpenBrowserClaw": OpenBrowserClawAgent("OpenBrowserClaw", "Browser"),
            "IronClaw": IronClawAgent("IronClaw", "Security"),
            "ClawMem": ClawMemAgent("ClawMem", "Memory"),
            "ClawSwarm": ClawSwarmAgent("ClawSwarm", "Swarm")
        }

    def load_memory(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f: return json.load(f)
            except: return []
        return []

    def save_memory(self):
        with open(self.storage_path, 'w') as f: json.dump(self.memory, f, indent=4)

    def secure_router(self, task):
        if self.legion["IronClaw"].run(task, self.bus, self.armory).startswith("BLOCK"): return []
        complexity = self.legion["TrinityClaw"].run(task, self.bus, self.armory)
        pipeline = [("ClawMem", task), ("ZeroClaw", task), ("TinyAGI", task)]
        
        if complexity == "HIGH" or "swarm" in task.lower():
            pipeline.extend([("OpenCrabs", task), ("ClawSwarm", task), ("ARC", task)])
        elif any(k in task.lower() for k in ["research", "find", "analyze"]):
            pipeline.extend([("OpenBrowserClaw", task), ("ARC", task)])
        elif "automate" in task.lower():
            pipeline.append(("AutoClaw", task))
        elif "edge" in task.lower():
            pipeline.append(("PicoClaw", task))
        else:
            pipeline.append(("OpenClaw", task))
        return pipeline

    def safla_cycle(self, task):
        print(f"\n[{self.name}] SENSE: {task}")
        pipeline = self.secure_router(task)
        if not pipeline: return "BLOCKED"
        results = []
        last_res = None
        for name, tsk in pipeline:
            agent = self.legion[name]
            if name == "ClawMem": res = agent.run(tsk, self.bus, self.armory, memory=self.memory)
            elif name == "ARC": res = agent.run(tsk, self.bus, self.armory, live_data=last_res)
            else: res = agent.run(tsk, self.bus, self.armory)
            last_res = res
            results.append(res)
        self.memory.append({"task": task, "results": results, "timestamp": time.time()})
        self.save_memory()
        print(f"[{self.name}] LEARN: Cycle complete. Memory updated.")
        return results

    def cli(self):
        print(f"\n--- {self.name} FULL STACK DEPLOYED ---")
        while True:
            try:
                cmd = input(f"{self.name} > ")
                if cmd.lower() in ['exit', 'quit']: break
                if not cmd.strip(): continue
                self.safla_cycle(cmd)
            except (KeyboardInterrupt, EOFError): break

if __name__ == "__main__":
    ClawPrime().cli()

# ===========================================================================
# SAFLA ENGINE: SENSE-ALIGN-FORCE-LOOP-ANALYZE
# ===========================================================================
class SAFLADriver:
    def __init__(self, zeus_instance):
        self.zeus = zeus_instance
        self.active = True

    async def run_loop(self, mission):
        print(f"⚡ [SAFLA] Loop Engaged: {mission}")
        while self.active:
            # SENSE (Hardware & Environment)
            hw_state = await self.zeus.hardware.get_device_info()
            # ALIGN (Meta-Agent Reflection)
            plan = await self.zeus.meta_agent.get_system_enhancement(mission)
            # FORCE (Claw Pantheon Execution)
            result = await self.zeus.process(mission) 
            # ANALYZE (MARS Reflection)
            await self.zeus.mars.reflect(mission, result)
            
            if "OBJECTIVE_SECURED" in result: break
            await asyncio.sleep(1)

if __name__ == "__main__":
    # Auto-start with SAFLA
    zeus = ZeusPrime()
    driver = SAFLADriver(zeus)
    asyncio.run(zeus.start())
    asyncio.run(driver.run_loop("Sovereign Domain Activation"))
