from dotenv import load_dotenv
import os
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, ToolMessage
from operator import add as add_messages
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool

load_dotenv()

llm = ChatOpenAI(model="gpt-4o", temperature=0) # Minimize hallucination

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

pdf_path = "/Users/mustaqode/Desktop/Langgraph_learn/mlbook.pdf"

if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

pdf_loader = PyPDFLoader(pdf_path)

try:
    pages = pdf_loader.load()
    print(f"PDF has been loaded and has {len(pages)} pages")
except Exception as e:
    print(f"Error loading PDF: {e}")
    raise

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000, 
    chunk_overlap = 200
)

pages_split = text_splitter.split_documents(pages)

persistent_dir = "/Users/mustaqode/Desktop/Langgraph_learn"
collection_name = "ml_learn"

if not os.path.exists(persistent_dir):
    os.makedirs(persistent_dir)

try:
    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persistent_dir,
        collection_name=collection_name
    )
    print("Chroma vector store is created!")
except Exception as e:
    print(f"Error setting up datatbase: {str(e)}")
    raise


retriever = vectorstore.as_retriever(
    search_type = "similarity",
    search_kwargs={"k": 5}
)

@tool
def retriever_tool(query: str) -> str:
    """This tool searches and returns the info about Machine Learning from the document"""
    
    docs = retriever.invoke(query)
    
    if not docs:
        return "I have no relevant information in the ML document"
    
    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i+1}:\n{doc.page_content}")
    
    return "\n\n".join(results)


tools = [retriever_tool]

llm = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    

def should_continue(state: AgentState):
    """Check if the last message contains a tool call"""
    result = state["messages"][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0


system_prompt = """
You are an intelligent AI assistant who can answer about machine learning based on the document provided (knowledge base).
Use the `retriever` tool available to answer your questions. 
You can make multiple calls if required.
You can look up some info before asking a follow up question if needed.
Please always cite the specific parts of the documents you use in your answers
NEVER answer anything outside the scope of the ML doc / knowledge base. Simply reply with a sorry message
"""

tools_dict = {our_tool.name : our_tool  for our_tool in tools }

# LLM Agent
def call_llm(state: AgentState) -> AgentState:
    """Function to call the LLM with the current state"""
    messages = list(state["messages"])
    messages = [SystemMessage(content=system_prompt)] + messages
    result = llm.invoke(messages)
    return {'messages': [result]}

# Retriever Agent [ Note that we don't use Toolnode to handle tool calls]
def take_action(state: AgentState) -> AgentState:
    """Execute tool calls from the LLM's response"""
    tool_calls = state["messages"][-1].tool_calls
    results = []
    
    for tool in tool_calls:
        print(f"Calling Tool:{tool['name']} with query: {tool['args'].get('query', 'No query provided')}")
        
        if not tool['name'] in tools_dict: #Checks if a valid tool is present
            print(f"\n Tool {tool['name']} doesn't exist")
            result = "Incorrect tool name, please retry and select tool from a list of available tools" 
        
        else:
            result = tools_dict[tool['name']].invoke(tool['args'].get('query', ''))
            print(f"Result length: {len(str(result))}")
        
        tool_call_id = tool['id']
        tool_name = tool['name']
        content = str(result)
        
        results.append(ToolMessage(tool_call_id = tool_call_id, name = tool_name, content=content))

    print("Tools Execution complete! Back to the model!")
    return {'messages': results}


graph = StateGraph(AgentState)

graph.add_node("llm_agent", call_llm)
graph.add_node("retriever_agent", take_action)
graph.set_entry_point("llm_agent")
graph.add_conditional_edges(
    "llm_agent",
    should_continue,
    {
        True: "retriever_agent",
        False: END
    }
)
graph.add_edge("retriever_agent", "llm_agent")

agent = graph.compile()

def run_rag_agent():
    print("\n === RAG AGENT ===")
    
    while True:
        user_input = input("\n What do you wanna learn in ML?\n")
        if user_input.lower() in ['exit', 'break']:
            break
        
        messages = [HumanMessage(content=user_input)]
        
        result = agent.invoke({'messages': messages})
        
        print("\n === ANSWER ===")
        print(result['messages'][-1].content)

if __name__ == '__main__':
    run_rag_agent()