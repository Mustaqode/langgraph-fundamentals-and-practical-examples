from typing import Annotated, TypedDict, Sequence
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import  ChatOpenAI
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode


load_dotenv()

## State

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

## Tools 

@tool
def add(a: int, b:int):
    """This is an addition function"""
    return a + b 

@tool
def sub(a: int, b:int):
    """This is a subtraction function"""
    return a - b 

@tool
def mult(a: int, b:int):
    """This is a multiplication function"""
    return a * b 

tools = [add, sub, mult]

##LLM init

llm = ChatOpenAI(model="gpt-4o").bind_tools(tools=tools)

## Nodes & Edges

def do_math(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=  "You are a math AI assistant. Answer politely in less than 50 words. "
            "Important: you MUST NOT compute numeric results yourself. For any arithmetic or numeric "
            "operation the user requests, you must call the provided tools (add/sub/mult) and only "
            "respond with tool_calls. Do not output numeric results or calculations in the assistant text. "
            "If you cannot use tools to satisfy the request, ask for clarification.")
    
    response = llm.invoke([system_prompt] + state["messages"])
    return {"messages": [response]} # Reducer function appends the new messages to the state
    
def should_continue(state: AgentState):
    messages = state["messages"]
    last_msg = messages[-1]
    
    if not last_msg.tool_calls:
        return "end"
    else:
        return "continue"
    
## Build graph

graph = StateGraph(AgentState)
tool_node = ToolNode(tools=tools)

graph.add_node("do_math", do_math)
graph.add_node("tool_node", tool_node)

graph.set_entry_point("do_math")

graph.add_conditional_edges(
    "do_math",
    should_continue,
    {
        "end" : END,
        "continue": "tool_node"
    }
)

graph.add_edge("tool_node", "do_math")

agent = graph.compile()


## Result

def print_steam(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()
            

user_input = input("Enter: ")

while user_input != "exit":
    inputs = {"messages": [("user", user_input)]} ## Tuple gets auto conversted to human message (Better to pass HumanMsg)
    print_steam(agent.stream(inputs, stream_mode="values"))    
    user_input = input("Enter: ")        

