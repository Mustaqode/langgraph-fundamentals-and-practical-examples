from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool 
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

load_dotenv()

# This is a global variable to store document content (Recomended: state injection to tools)
document_content = ""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    
@tool
def update(content: str) -> str:
    """Updates the document with the provided content"""
    global document_content
    document_content = content
    return f"Document has been updated successfully! The current content is: \n{document_content}"

@tool
def save(filename: str) -> str:
    """ Save the current document to a text file and finish the process.
    
    Args:
        filename: Name for the text file
    """
    
    global document_content
    
    if not filename.endswith('.txt'):
        filename = f"{filename}.txt"
        
    try:
        with open(filename, 'w') as file:
            file.write(document_content)
        print(f"\n Document has been saved to {filename}")
        return f"Document has been saved successfully to `{filename}`."
    except Exception as e:
        return f"Error saving document: {str(e)}"
    
tools = [update, save]

llm = ChatOpenAI(model='gpt-4o').bind_tools(tools=tools)


def drafter_agent(state: AgentState) -> AgentState:
    system_prompt = f"""
    You are a drafter, a helpful writing assistant. You are gonna help user update and save the document.
    - If the user wants or modify content, use the `update` tool with completed document
    - If the user wants to save and finish, you must use `save` tool.
    - Make sure to always show the current document state after modifications
    
    The current document content is: {document_content}
    """
    
    system_msg = SystemMessage(content=system_prompt)
    
    if not state["messages"]:
        user_input = input("\nI'm ready to help you update a document. What do you want to create?\n")
        user_message = HumanMessage(content=user_input)
    
    else:
        user_input = input("\n What would you like to do with the doc? \n")
        print(f"\n USER: {user_input}")
        user_message = HumanMessage(content=user_input)
    
    all_messages = [system_msg] + list(state["messages"]) + [user_message]
    
    response = llm.invoke(all_messages)
    
    print(f"\n AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"USING TOOLS: { [tc['name'] for tc in response.tool_calls] }")
        
    return {"messages": list(state['messages']) + [user_message, response] }

def should_continue(state: AgentState): 
    """Determine if we should continue or end the convo"""
    
    messages = state["messages"]
    
    if not messages:
        return "continue"
    
    # Looks for recent tool message...
    for message in reversed(messages):
        #... and checks if this is a ToolMessage from save
        if(isinstance(message, ToolMessage) and 
            "saved" in message.content.lower() and 
            "document" in message.content.lower()):
            return "end" # goes to end edge
        else:
            return "continue"
    
def print_messages(messages) :
   """Function I made to print the messages in a more readable format""" 
   if not messages: 
       return
   
   for message in messages[-3:]: 
       if isinstance(message, ToolMessage) :
           print(f"\n*TOOL RESULT: {message.content}")   


graph = StateGraph(AgentState)

graph.add_node("agent", drafter_agent)
graph.add_node("tool_node", ToolNode(tools))

graph.set_entry_point("agent")

graph.add_edge("agent", "tool_node")

graph.add_conditional_edges(
    "tool_node",
    should_continue,
    {
        "continue" : "agent",
        "end": END
    }
)

agent = graph.compile()

def run_drafter_agent():
    print("\n ===== DRAFTER =====")
    
    state = {"messages": []}
    
    for step in agent.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
            
    print("\n ===== DRAFTER FINISHED =====")


if __name__ == "__main__":
    run_drafter_agent() 