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