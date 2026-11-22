"""Cross-platform Windows <-> WSL2 path helpers and display formatting."""

from __future__ import annotations

import os
import platform
import re
from functools import lru_cache
from pathlib import Path

_WIN_DRIVE_RE = re.compile(r"^([A-Za-z]):[\\/](.*)$")
_WIN_DRIVE_ROOT_RE = re.compile(r"^([A-Za-z]):$")
_WSL_MNT_RE = re.compile(r"^/mnt/([a-zA-Z])(?:/(.*))?$")


@lru_cache(maxsize=1)
def is_wsl() -> bool:
    """Return True when running inside Windows Subsystem for Linux (WSL/WSL2)."""
    if os.name != "posix":
        return False
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    rel = platform.release().lower()
    if "microsoft" in rel or "wsl" in rel:
        return True
    try:
        return "microsoft" in Path("/proc/version").read_text(encoding="utf-8", errors="ignore").lower()
    except OSError:
        return False


def win_to_wsl_path(raw: str) -> str:
    """Convert a Windows drive path (e.g. 'D:\\ai\\packs') into '/mnt/d/ai/packs'."""
    text = raw.strip()
    m_root = _WIN_DRIVE_ROOT_RE.match(text)
    if m_root:
        return f"/mnt/{m_root.group(1).lower()}"
    m = _WIN_DRIVE_RE.match(text)
    if not m:
        return raw
    drive = m.group(1).lower()
    rest = m.group(2).replace("\\", "/").lstrip("/")
    return f"/mnt/{drive}/{rest}" if rest else f"/mnt/{drive}"


def wsl_to_win_path(raw: str) -> str:
    """Convert a WSL '/mnt/<drive>/...' path into Windows 'D:\\...' format."""
    text = raw.strip()
    m = _WSL_MNT_RE.match(text)
    if not m:
        return raw
    drive = m.group(1).upper()
    rest = (m.group(2) or "").replace("/", "\\")
    return f"{drive}:\\{rest}" if rest else f"{drive}:\\"


def normalize_user_path(raw: str | Path, base_dir: Path | None = None) -> Path:
    """Normalize a path entered in Windows, WSL2, or POSIX format to a resolved runtime Path."""
    text = str(raw).strip()
    if not text:
        return (base_dir or Path.cwd()).resolve()

    if is_wsl():
        text = win_to_wsl_path(text)
    elif os.name == "nt":
        m = _WSL_MNT_RE.match(text)
        if m:
            text = wsl_to_win_path(text)

    candidate = Path(text).expanduser()
    if not candidate.is_absolute() and base_dir is not None:
        candidate = base_dir / candidate
    return candidate.resolve()


def format_display_path(path: str | Path) -> str:
    """Format a runtime path cleanly for UI display (Windows 'D:\\...' on Windows and WSL2)."""
    text = str(path)
    if os.name == "nt":
        if _WSL_MNT_RE.match(text):
            return wsl_to_win_path(text)
        return str(Path(text))
    if is_wsl():
        if _WIN_DRIVE_RE.match(text) or _WIN_DRIVE_ROOT_RE.match(text):
            return text.replace("/", "\\")
        if _WSL_MNT_RE.match(text):
            return wsl_to_win_path(text)
    return text
