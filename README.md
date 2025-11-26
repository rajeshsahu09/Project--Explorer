# Project--Explorer

Small workspace demonstrating an LLM-driven agent workflow using LangChain-style tools and a filesystem backend.

Contents
- `app.py` — Example state graph that calls a chat model and executes tools.
- `llm_integration/llm_client.py` — A lightweight OpenLLM wrapper that calls Groq and normalizes tool-call output for LangChain message objects.
- `llm_tools/filesystem_tools.py` — Original filesystem helper functions (utility API).
- `llm_tools/filesystem_tool.py` — Wrapper added to expose `deepagents.backends.filesystem.FilesystemBackend` as LangChain `Tool`s (uses the provided `FilesystemBackend`, not a custom reimplementation).
- `aienv/` — Included Python virtual environment used for running the project.
- `aienv/Lib/site-packages/.../deepagents/backends/filesystem.py` — Provided `FilesystemBackend` implementation that the tool wrappers call.

Quickstart (Windows PowerShell)

1. Activate the existing virtual environment (provided in `aienv`):

```powershell
# If using the provided venv
.\aienv\Scripts\Activate.ps1

# Alternatively create and activate a new venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. (Optional) Install any missing dependencies in the active environment. The workspace already contains many packages in `aienv`; if you created a fresh venv, install required packages with:

```powershell
pip install -r requirements.txt  # if you create one, or install langchain-core, groq, etc.
```

3. Run the example app:

```powershell
python .\app.py
```

Notes / Troubleshooting

- AIMessage validation errors: The code expects `tool_call.args` to be a dictionary. If your LLM outputs arguments as a JSON string or plain string, the wrapper in `llm_integration/llm_client.py` attempts to coerce the model output into a dict (parsing JSON when possible and otherwise wrapping the text under an `input` key). If you still see validation errors, check the printed `response_text` and the traceback to see what form the model returned.

- Tool invocation: LangChain `Tool` objects accept either raw input or a full `ToolCall` object. When invoking tools from code (see `app.py`), prefer passing the full ToolCall dict (including `"type": "tool_call"` and `"id"`) so `BaseTool` can inject `tool_call_id` and run schema validation. Example:

```python
# preferred
tool.invoke({"name": "fs_read_file", "args": {"path": "foo.txt"}, "id": "...", "type": "tool_call"})

# avoid passing just args when tool expects injected fields
tool.invoke({"path": "foo.txt"})
```

- FilesystemBackend behavior: If you need sandboxed (virtual) behavior, construct `FilesystemBackend(root_dir=..., virtual_mode=True)` and pass it into `llm_tools.filesystem_tool.get_langchain_tools(backend=your_backend)`.

Next steps you might want
- Run `app.py` and paste any tracebacks here so I can help debug.
- Add a `requirements.txt` or `pyproject.toml` if you want reproducible installs.
- Add unit tests for `llm_tools/filesystem_tool.py`.

If you want, I can run the app now and share the output (I can also add a minimal `requirements.txt`).

**Sample Run Output**

Below is a captured run of `app.py` (trimmed slightly where very large file contents were printed). This shows the agent requesting file tools, searching for `llm_client.py`, and returning the file contents when found under `llm_integration/`.

```
================================ Human Message =================================

summarise the llm_client.py file
================================== Ai Message ==================================
Tool Calls:
	fs_read_file (gwkwx081g)
 Call ID: gwkwx081g
	Args:
		__arg1: llm_client.py
Invoking tool: {'name': 'fs_read_file', 'args': {'__arg1': 'llm_client.py'}, 'id': 'gwkwx081g', 'type': 'tool_call'}
================================= Tool Message =================================
Name: fs_read_file

{"error": "D:\\AI\\tool_calling\\llm_client.py"}
================================== Ai Message ==================================

It seems there was an issue locating the `llm_client.py` file. Let me check the current dir
ectory structure to help diagnose this:
Tool Calls:
	fs_list_dir (m1bk60hb1)
 Call ID: m1bk60hb1
	Args:
		__arg1: .
Invoking tool: {'name': 'fs_list_dir', 'args': {'__arg1': '.'}, 'id': 'm1bk60hb1', 'type': 'tool_call'}
================================= Tool Message =================================
Name: fs_list_dir

[".git", ".gitignore", ".pytest_cache", "agent.png", "aienv", "app.py", "llm_integration", "llm_tools", "README.md", "__pycache__"]
================================== Ai Message ==================================

The `llm_client.py` file doesn't appear to exist in the root directory. Let me search for i
t recursively in subdirectories:
Tool Calls:
	fs_search_files (25cdt1t9a)
 Call ID: 25cdt1t9a
	Args:
		__arg1: llm_client.py
Invoking tool: {'name': 'fs_search_files', 'args': {'__arg1': 'llm_client.py'}, 'id': '25cdt1t9a', 'type': 'tool_call'}
================================= Tool Message =================================
Name: fs_search_files

["llm_integration\\llm_client.py"]
================================== Ai Message ==================================
Tool Calls:
  fs_read_file (7k73tb2tw)
 Call ID: 7k73tb2tw
  Args:
	__arg1: llm_integration\llm_client.py
Invoking tool: {'name': 'fs_read_file', 'args': {'__arg1': 'llm_integration\\llm_client.py'}, 'id': '7k73tb2tw', 'type': 'tool_call'}
================================= Tool Message =================================
Name: fs_read_file

```python
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages.base import BaseMessage
from langchain_core.messages.ai import AIMessage
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import ChatResult
from langchain_core.outputs.chat_generation import ChatGeneration
from langchain_core.tools.base import BaseTool
from langchain_core.runnables.base import Runnable
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain.messages import ToolCall
import os
from groq import Groq
import typing
from typing import Any, Sequence, Callable, Optional, List, Dict
import sys
sys.stderr = open(os.devnull, 'w')

class LLMClient:
	"""A simple LLM provider interface."""
	def __init__(self):
		self.api_key = os.environ.get('GROQ_API_KEY')
		self.client = Groq(api_key=self.api_key)
		self.model = "qwen/qwen3-32b" #'llama-3.1-8b-instant' #"llama-3.3-70b-versatile"

	def generate_response(self, prompt: str) -> Any:
		"""Generate a response from the LLM."""
		chat_completion = self.client.chat.completions.create(
			messages=[
				{
					"role": "user",
					"content": prompt
				}
			],
			model=self.model,
			include_reasoning=False,
		)
		return chat_completion.choices[0].message.content

class OpenLLM(BaseChatModel):
	llm: LLMClient = LLMClient()
	bound_tools: Optional[List[Dict]] = None

	def _generate(
		self,
		messages: list[BaseMessage],
		stop: list[str] | None = None,
		run_manager: CallbackManagerForLLMRun | None = None,
		**kwargs: Any,
	) -> ChatResult:
		"""Generate the result.

		Args:
			messages: The messages to generate from.
			stop: Optional list of stop words to use when generating.
			run_manager: Optional callback manager to use for this call.
			**kwargs: Additional keyword arguments to pass to the model.

		Returns:
			The chat result.
		"""
		system_instruction = ""
		if self.bound_tools:
			tools_json = [t["function"] for t in self.bound_tools]
			system_instruction = (
				f"\nYou have access to the following tools:\n{tools_json}\n"
				"If you need to use a tool, output ONLY a JSON object with 'name' and 'arguments'."
			)

		prompt = messages[-1].content + system_instruction
		response_text = self.llm.generate_response(prompt)

		message_kwargs = {}
		tool_calls = []
        
		try:
			# Simple check: assumes model outputs pure JSON for tools
			import json, uuid
			if "{" in response_text and "}" in response_text:
				response_text = response_text.split('json')[-1].split('```')[0]
				data = json.loads(response_text)
				if "name" in data and "arguments" in data:
					tool_calls.append({
						"name":data["name"],
						"args":data["arguments"],
						"id":uuid.uuid4().hex,
						"type": "tool_call"
					})
		except Exception:
			pass
		final_message = AIMessage(
			content=response_text if not tool_calls else "",
			tool_calls=tool_calls,
		)
		generation = ChatGeneration(message=final_message)
		return ChatResult(generations=[generation])
    
	def _llm_type(self) -> str:
		return "openllm"
    
	def bind_tools(
		self,
		tools: Sequence[
			typing.Dict[str, Any] | type | Callable | BaseTool  # noqa: UP006
		],
		*,
		tool_choice: str | None = None,
		**kwargs: Any,
	) -> Runnable[LanguageModelInput, AIMessage]:
		"""Bind tools to the model.

		Args:
			tools: Sequence of tools to bind to the model.
			tool_choice: The tool to use. If "any" then any tool can be used.

		Returns:
			A Runnable that returns a message.

		"""
		formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        
		# Return a new instance of your class with the tools attached
		return self.__class__(bound_tools=formatted_tools, **self.dict())
================================== Ai Message ==================================

Here's a summary of the `llm_integration/llm_client.py` file:

---

### **Key Components**

1. **LLMClient Class**
   - **Purpose**: Simplified interface for interacting with a Groq-hosted LLM (Large Language Model).
   - **Initialization**:
     - Uses an API key from the environment (`GROQ_API_KEY`).
     - Configures the model as `"qwen/qwen3-32b"` by default.
   - **Core Method**:
     - `generate_response(prompt: str) -> str`: Sends a user prompt to the LLM via the Groq client and returns the model's response.

2. **OpenLLM Class**
   - **Purpose**: Integrates the Groq client with LangChain, enabling tool usage and chat model compatibility.
   - **Inheritance**: Extends `BaseChatModel` from `langchain_core`.
   - **Key Features**:
     - **Tool Binding**:
       - Supports attaching tools (e.g., functions, APIs) via `bind_tools()`.
       - Converts tools into OpenAI-compatible format using `convert_to_openai_tool()`.    
     - **Response Handling**:
       - Generates responses using the `LLMClient`.
       - Parses for structured **tool calls** in the LLM's output (e.g., JSON objects with `name` and `arguments`).
       - Returns responses as `ChatResult` objects with `AIMessage` content.

---

### **Key Dependencies**
- **Groq API**: For LLM inference (`groq.Groq`).
- **LangChain Core**: For integration with LangChain's abstractions (`BaseChatModel`, `ChatResult`, `AIMessage`, etc.).
- **Utilities**: Uses `os`, `json`, `uuid`, and `sys` for environment, parsing, and logging.

---

### **Usage Example**
```python
from llm_integration.llm_client import LLMClient, OpenLLM

# Basic LLM inference
client = LLMClient()
response = client.generate_response("Write a haiku about AI.")
print(response)

# With LangChain integration and tools
llm = OpenLLM()
result = llm.generate([HumanMessage(content="What's the weather in Paris?")])
print(result.generations[0].message.content)
```

---

### **Notes**
- The `sys.stderr` is redirected to `os.devnull` to suppress Groq client logging.
- Tool calls are extracted via a **simple heuristic** (checks for JSON-like syntax in responses).

Let me know if you'd like a deeper dive into any specific part!
```
```

If you want the full untrimmed output or the exact file content pasted into the README, tell me and I will expand this section.