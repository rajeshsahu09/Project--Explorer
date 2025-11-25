"""Example showing how to bind filesystem tools to a LangChain chat agent.

Run this with the workspace virtualenv Python. It demonstrates listing files
and searching for occurrences of a query via the tool wrappers.
"""
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path when running the script directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from llm_tools.filesystem_tools import get_langchain_tools, read_file


def main():
    try:
        tools = get_langchain_tools()
        print("Created tools:", [t.name for t in tools])
    except Exception as e:
        print("Could not create LangChain tools:", e)
        print("Proceeding with direct function demo (install langchain to create tool wrappers).")
        tools = []

    # Simple direct call demonstration (without LangChain agent):
    print("\nListing root (non-recursive):")
    from llm_tools.filesystem_tools import list_dir

    for p in list_dir('.', recursive=False):
        print(" -", p)

    # Demonstrate find_in_files by searching for the word 'TODO'
    from llm_tools.filesystem_tools import find_in_files

    print('\nSearching for "TODO" in project files...')
    matches = find_in_files('TODO', max_results=10)
    for m in matches[:20]:
        print(f"{m.path}:{m.line_no}: {m.line}")


if __name__ == '__main__':
    main()
