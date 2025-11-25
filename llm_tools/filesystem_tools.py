"""Filesystem tools for use with LLM agents (LangChain-compatible).

This module provides safe file access helpers plus an opinionated wrapper
to create LangChain `Tool`s. It includes:
- configuration (allow/deny lists)
- `set_base_dir()` to scope operations during tests
- chunked reads and a text chunker for large files

Security: operations are constrained to the configured `BASE_DIR` and can be
further restricted with allow/deny lists plus allowed/denied extensions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Generator, List, Optional


# Default project base; can be changed with `set_base_dir` (useful for tests)
BASE_DIR = Path.cwd()


def set_base_dir(path: str) -> None:
    """Set the module base directory used for path resolution (mostly for tests)."""
    global BASE_DIR
    BASE_DIR = Path(path).resolve()


@dataclass
class FSConfig:
    # Paths are relative to BASE_DIR
    allowed_paths: Optional[List[str]] = None  # if None => all under BASE_DIR allowed
    denied_paths: Optional[List[str]] = None
    allowed_extensions: Optional[List[str]] = None  # like ['.py', '.md']
    denied_extensions: Optional[List[str]] = None
    max_read_chars: int = 20000


def _resolve_within_base(path: str) -> Path:
    p = (BASE_DIR / Path(path)).resolve()
    base = BASE_DIR.resolve()
    try:
        p.relative_to(base)
    except Exception:
        raise ValueError(f"Path {p} is outside the allowed base directory {base}")
    return p


def _matches_path_list(p: Path, patterns: Optional[List[str]]) -> bool:
    if not patterns:
        return False
    rel = p.relative_to(BASE_DIR)
    for pat in patterns:
        candidate = (BASE_DIR / pat).resolve()
        try:
            p.relative_to(candidate)
            return True
        except Exception:
            # not a subpath
            continue
    return False


def _is_extension_allowed(p: Path, config: Optional[FSConfig]) -> bool:
    if not config:
        return True
    if config.allowed_extensions is not None:
        return p.suffix in config.allowed_extensions
    if config.denied_extensions is not None:
        return p.suffix not in config.denied_extensions
    return True


def _is_path_allowed(p: Path, config: Optional[FSConfig]) -> bool:
    # Must be within base dir
    try:
        p.relative_to(BASE_DIR)
    except Exception:
        return False
    if config is None:
        return True
    if config.allowed_paths:
        if not _matches_path_list(p, config.allowed_paths):
            return False
    if config.denied_paths:
        if _matches_path_list(p, config.denied_paths):
            return False
    if not _is_extension_allowed(p, config):
        return False
    return True


def read_file(path: str, max_chars: Optional[int] = None, encoding: str = "utf-8", config: Optional[FSConfig] = None) -> str:
    """Read file contents, truncated to config.max_read_chars or `max_chars` if provided."""
    p = _resolve_within_base(path)
    if not _is_path_allowed(p, config):
        raise PermissionError(f"Access to {p} is denied by FSConfig")
    if not p.exists():
        raise FileNotFoundError(str(p))
    text = p.read_text(encoding=encoding)
    limit = max_chars if max_chars is not None else (config.max_read_chars if config else 20000)
    if limit and len(text) > limit:
        return text[:limit] + "\n...[truncated]"
    return text


def read_file_chunked(path: str, chunk_size: int = 4000, overlap: int = 200, encoding: str = "utf-8", config: Optional[FSConfig] = None) -> List[str]:
    """Return a list of text chunks for the given file.

    Each chunk will be at most `chunk_size` characters and consecutive chunks
    overlap by `overlap` characters to preserve context.
    """
    text = read_file(path, max_chars=None, encoding=encoding, config=config)
    return text_chunk(text, chunk_size=chunk_size, overlap=overlap)


def text_chunk(text: str, chunk_size: int = 4000, overlap: int = 200) -> List[str]:
    """Chunk a string into overlapping pieces.

    Returns a list of chunks. Last chunk may be shorter.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks: List[str] = []
    start = 0
    L = len(text)
    while start < L:
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= L:
            break
        start = end - overlap
    return chunks


def write_file(path: str, content: str, overwrite: bool = True, encoding: str = "utf-8", config: Optional[FSConfig] = None) -> str:
    p = _resolve_within_base(path)
    if not _is_path_allowed(p, config):
        raise PermissionError(f"Access to {p} is denied by FSConfig")
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists() and not overwrite:
        raise FileExistsError(str(p))
    p.write_text(content, encoding=encoding)
    return f"Wrote {len(content)} bytes to {p}"


def append_file(path: str, content: str, encoding: str = "utf-8", config: Optional[FSConfig] = None) -> str:
    p = _resolve_within_base(path)
    if not _is_path_allowed(p, config):
        raise PermissionError(f"Access to {p} is denied by FSConfig")
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding=encoding) as f:
        f.write(content)
    return f"Appended {len(content)} bytes to {p}"


def edit_file(path: str, find: str, replace: str, max_replacements: Optional[int] = None, use_regex: bool = False, encoding: str = "utf-8", config: Optional[FSConfig] = None) -> str:
    p = _resolve_within_base(path)
    if not _is_path_allowed(p, config):
        raise PermissionError(f"Access to {p} is denied by FSConfig")
    if not p.exists():
        raise FileNotFoundError(str(p))
    text = p.read_text(encoding=encoding)
    if use_regex:
        new_text, n = re.subn(find, replace, text, count=0 if max_replacements is None else max_replacements)
    else:
        if max_replacements is None:
            new_text = text.replace(find, replace)
            n = text.count(find)
        else:
            new_text = text.replace(find, replace, max_replacements)
            n = sum(1 for _ in re.finditer(re.escape(find), text))
            n = min(n, max_replacements)
    p.write_text(new_text, encoding=encoding)
    return f"Replaced {n} occurrences in {p}"


def list_dir(path: str = ".", recursive: bool = False, config: Optional[FSConfig] = None) -> List[str]:
    p = _resolve_within_base(path)
    if not _is_path_allowed(p, config):
        raise PermissionError(f"Access to {p} is denied by FSConfig")
    if not p.exists():
        raise FileNotFoundError(str(p))
    if p.is_file():
        return [str(p)]
    if recursive:
        return [str(x.relative_to(BASE_DIR)) for x in p.rglob("*")]
    else:
        return [str(x.relative_to(BASE_DIR)) for x in p.iterdir()]


def search_files(pattern: str = "*", root: str = ".", file_glob: str = "**/*", case_sensitive: bool = False, config: Optional[FSConfig] = None) -> List[str]:
    r = _resolve_within_base(root)
    if not _is_path_allowed(r, config):
        raise PermissionError(f"Access to {r} is denied by FSConfig")
    files = [f for f in r.rglob(file_glob) if f.is_file()]
    if not case_sensitive:
        pattern = pattern.lower()
        files = [f for f in files if pattern in f.name.lower()]
    else:
        files = [f for f in files if pattern in f.name]
    # apply extension filtering
    if config and config.allowed_extensions is not None:
        files = [f for f in files if f.suffix in config.allowed_extensions]
    if config and config.denied_extensions is not None:
        files = [f for f in files if f.suffix not in config.denied_extensions]
    return [str(f.relative_to(BASE_DIR)) for f in files]


@dataclass
class Match:
    path: str
    line_no: int
    line: str


def find_in_files(query: str, root: str = ".", file_glob: str = "**/*", use_regex: bool = False, case_sensitive: bool = False, max_results: Optional[int] = 200, config: Optional[FSConfig] = None) -> List[Match]:
    r = _resolve_within_base(root)
    if not _is_path_allowed(r, config):
        raise PermissionError(f"Access to {r} is denied by FSConfig")
    files = [f for f in r.rglob(file_glob) if f.is_file()]
    matches: List[Match] = []
    flags = 0 if case_sensitive else re.IGNORECASE
    pattern = re.compile(query, flags) if use_regex else None
    for f in files:
        if not _is_path_allowed(f, config):
            continue
        try:
            text = f.read_text(errors="ignore")
        except Exception:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            found = False
            if use_regex:
                if pattern.search(line):
                    found = True
            else:
                if case_sensitive:
                    found = query in line
                else:
                    found = query.lower() in line.lower()
            if found:
                matches.append(Match(path=str(f.relative_to(BASE_DIR)), line_no=i, line=line.strip()))
                if max_results is not None and len(matches) >= max_results:
                    return matches
    return matches


def file_info(path: str, config: Optional[FSConfig] = None) -> dict:
    p = _resolve_within_base(path)
    if not _is_path_allowed(p, config):
        raise PermissionError(f"Access to {p} is denied by FSConfig")
    st = p.stat()
    return {"path": str(p.relative_to(BASE_DIR)), "size": st.st_size, "is_file": p.is_file(), "is_dir": p.is_dir()}


def get_langchain_tools(prefix: str = "fs", config: Optional[FSConfig] = None):
    """Return a list of LangChain `Tool` objects wrapping the functions.

    Each tool is a thin wrapper that enforces the provided `config` at call time.
    """
    try:
        from langchain_core.tools.simple import Tool
    except Exception as e:
        raise ImportError("langchain is required to build Tool wrappers. Install with `pip install langchain`.") from e

    # wrappers to bind config
    def _wrap(fn):
        def inner(*args, **kwargs):
            kwargs.setdefault("config", config)
            return fn(*args, **kwargs)

        inner.__name__ = fn.__name__
        return inner

    return [
        Tool.from_function(_wrap(read_file), name=f"{prefix}_read_file", description="Read a file from the project and return its contents"),
        Tool.from_function(_wrap(write_file), name=f"{prefix}_write_file", description="Write contents to a file in the project"),
        Tool.from_function(_wrap(append_file), name=f"{prefix}_append_file", description="Append contents to a file in the project"),
        Tool.from_function(_wrap(edit_file), name=f"{prefix}_edit_file", description="Edit a file by replacing text or regex"),
        Tool.from_function(_wrap(list_dir), name=f"{prefix}_list_dir", description="List files/directories in a path"),
        Tool.from_function(_wrap(search_files), name=f"{prefix}_search_files", description="Search for files by name pattern"),
        Tool.from_function(_wrap(find_in_files), name=f"{prefix}_find_in_files", description="Search file contents and return matching lines"),
        Tool.from_function(_wrap(file_info), name=f"{prefix}_file_info", description="Return basic file info (size, type)"),
        Tool.from_function(_wrap(read_file_chunked), name=f"{prefix}_read_file_chunked", description="Read a file and return text chunks for streaming large files"),
    ]


__all__ = [
    "FSConfig",
    "set_base_dir",
    "read_file",
    "read_file_chunked",
    "text_chunk",
    "write_file",
    "append_file",
    "edit_file",
    "list_dir",
    "search_files",
    "find_in_files",
    "file_info",
    "get_langchain_tools",
    "Match",
]
