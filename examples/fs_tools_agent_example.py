"""Example showing how to bind filesystem tools to a LangChain chat agent.

This demonstrates creating an `FSConfig` allowlist and building Tool wrappers
with `get_langchain_tools`. To run the agent you need an LLM provider and
`langchain` installed. The example below uses `ChatOpenAI` (OpenAI) — set
`OPENAI_API_KEY` in your environment before running.
"""
import os
import sys
from pathlib import Path

# ensure project root on sys.path when executing the example
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from llm_tools.filesystem_tools import FSConfig, get_langchain_tools


def main():
    # Restrict agent to only see `examples` and `llm_tools` directories
    config = FSConfig(allowed_paths=["examples", "llm_tools"], allowed_extensions=[".py", ".md"], max_read_chars=10000)

    try:
        tools = get_langchain_tools(prefix="fs", config=config)
    except Exception as e:
        print("Install langchain to run this agent example:", e)
        return

    # Example agent setup (you must install langchain and a chat model SDK)
    try:
        from langchain.chat_models import ChatOpenAI
        from langchain.agents import initialize_agent, AgentType
    except Exception as e:
        print("LangChain or chat model dependencies not installed:", e)
        return

    if not os.environ.get("OPENAI_API_KEY"):
        print("Set OPENAI_API_KEY in your environment to run the agent example.")
        return

    model = ChatOpenAI(temperature=0)
    agent = initialize_agent(tools, model, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

    # Example prompt: ask the agent to list python files in allowed directories
    prompt = "List Python files the agent can access and show the first line of each."
    result = agent.run(prompt)
    print("Agent result:\n", result)


if __name__ == '__main__':
    main()
