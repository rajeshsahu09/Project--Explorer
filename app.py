from typing import (
    Annotated,
    Sequence,
    TypedDict,
)
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """The state of the agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]

from llm_integration.llm_client import OpenLLM
from llm_tools import filesystem_tools

tools = filesystem_tools.get_langchain_tools()
model = OpenLLM()

from langchain_groq import ChatGroq
model = ChatGroq(
    model_name="qwen/qwen3-32b", #'llama-3.1-8b-instant', #"llama-3.3-70b-versatile",
    temperature=0.7
)

model = model.bind_tools(tools=tools)

import json
from langchain_core.messages import ToolMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

tools_by_name = {tool.name: tool for tool in tools}

# Define our tool node
def tool_node(state: AgentState):
    outputs = []
    for tool_call in state["messages"][-1].tool_calls:
        print("Invoking tool:", tool_call)
        tool = tools_by_name.get(tool_call["name"])
        try:
            tool_result = tool.invoke(tool_call["args"])
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            tool_result = {"error": str(e)}

        # If the tool already returned a ToolMessage, forward it directly.
        if isinstance(tool_result, ToolMessage):
            outputs.append(tool_result)
        else:
            # Ensure content is serializable JSON or a string
            try:
                content = json.dumps(tool_result)
            except Exception:
                content = str(tool_result)
            outputs.append(
                ToolMessage(
                    content=content,
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
    return {"messages": outputs}


# Define the node that calls the model
def call_model(
    state: AgentState,
    config: RunnableConfig,
):
    # this is similar to customizing the create_react_agent with 'prompt' parameter, but is more flexible
    system_prompt = SystemMessage(
        "You are a helpful AI assistant, please respond to the users query to the best of your ability!"
    )
    response = model.invoke([system_prompt] + state["messages"], config)
    return {"messages": [response]}


# Define the conditional edge that determines whether to continue or not
def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "continue"
    else:
        return "end"
    
from langgraph.graph import StateGraph, END

# Define a new graph
workflow = StateGraph(AgentState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)
workflow.add_edge("tools", "agent")

graph = workflow.compile()
# print(graph.get_graph().draw_mermaid())

# Helper function for formatting the stream nicely
def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()


inputs = {"messages": [("user", "summarise the llm_client.py file")]}
print_stream(graph.stream(inputs, stream_mode="values"))