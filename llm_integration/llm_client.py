from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages.base import BaseMessage
from langchain_core.messages.ai import AIMessage
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.outputs import ChatResult
from langchain_core.outputs.chat_generation import ChatGeneration
from typing import Any
import os
from groq import Groq
import sys
sys.stderr = open(os.devnull, 'w')

class LLMClient:
    """A simple LLM provider interface."""
    def __init__(self):
        self.api_key = os.environ.get('GROQ_API_KEY')
        self.client = Groq(api_key=self.api_key)
        self.model = "llama-3.3-70b-versatile"

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
        )
        return chat_completion.choices[0].message.content

class OpenLLM(BaseChatModel):
    llm: LLMClient = LLMClient()

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
        prompt = "\n".join([message.content for message in messages])
        response_text = self.llm.generate_response(prompt)
        generation = ChatGeneration(message=AIMessage(content=response_text))
        return ChatResult(generations=[generation])
    
    def _llm_type(self) -> str:
        return "openllm"
    