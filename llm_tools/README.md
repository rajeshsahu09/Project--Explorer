# Filesystem LLM Tools

This package provides a set of safe filesystem helper functions and a helper
to wrap them as LangChain `Tool`s so a BaseChatModel/agent can interact with
the project files.

Key functions
- `read_file(path, max_chars=...)` — read file contents (truncated)
- `write_file(path, content, overwrite=True)` — write content
- `append_file(path, content)` — append content
- `edit_file(path, find, replace, ...)` — find/replace (supports regex)
- `list_dir(path, recursive=False)` — list files
- `search_files(pattern, root, file_glob)` — search filenames
- `find_in_files(query, ...)` — search file contents, returns matches
- `file_info(path)` — basic file metadata
- `get_langchain_tools(prefix='fs')` — returns a list of LangChain `Tool`s

Security
- All file paths are resolved relative to the process working directory (`Path.cwd()`).
- Attempts to access files outside this base directory will raise a `ValueError`.

Usage example
See `examples/fs_tools_example.py` for a runnable demonstration that binds
the tools to a LangChain chat model (agent).
