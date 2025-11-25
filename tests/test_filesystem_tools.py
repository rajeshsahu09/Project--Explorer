import os
import tempfile
from pathlib import Path

import pytest

from llm_tools.filesystem_tools import (
    FSConfig,
    set_base_dir,
    read_file,
    write_file,
    text_chunk,
    read_file_chunked,
    list_dir,
)


def test_write_and_read(tmp_path):
    set_base_dir(str(tmp_path))
    p = tmp_path / "sub"
    p.mkdir()
    fp = p / "hello.txt"
    write_file(str(fp.relative_to(tmp_path)), "hello world")
    content = read_file(str(fp.relative_to(tmp_path)))
    assert "hello world" in content


def test_chunking():
    text = "".join(str(i) for i in range(1000))
    chunks = text_chunk(text, chunk_size=100, overlap=10)
    # Ensure chunks cover the text
    assert "".join(chunks).startswith(text[:100])
    assert sum(len(c) for c in chunks) >= len(text)


def test_read_file_chunked(tmp_path):
    set_base_dir(str(tmp_path))
    fp = tmp_path / "big.txt"
    data = "0123456789" * 1000
    fp.write_text(data)
    chunks = read_file_chunked(str(fp.relative_to(tmp_path)), chunk_size=500, overlap=50)
    assert len(chunks) > 1


def test_allow_deny(tmp_path):
    # create allowed and denied dirs
    a = tmp_path / "allowed"
    d = tmp_path / "denied"
    a.mkdir()
    d.mkdir()
    af = a / "a.txt"
    df = d / "d.txt"
    af.write_text("A")
    df.write_text("D")
    set_base_dir(str(tmp_path))
    cfg = FSConfig(allowed_paths=["allowed"], denied_paths=["denied"])
    # allowed file should be readable
    s = read_file(str(af.relative_to(tmp_path)), config=cfg)
    assert s == "A"
    # denied file should raise
    with pytest.raises(PermissionError):
        read_file(str(df.relative_to(tmp_path)), config=cfg)
