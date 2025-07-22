from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.graph import MessagesState, StateGraph
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import END
from langgraph.prebuilt import ToolNode, tools_condition
import sys
import os

# Add the parent directory to sys.path to import vector modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from vector.search import SemanticSearcher
except ImportError:
    # Fallback import for when running from different contexts
    import sys
    import os
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    from vector.search import SemanticSearcher

from dotenv import load_dotenv
load_dotenv(override=True)

# Initialize the semantic searcher with error handling
try:
    searcher = SemanticSearcher()
except Exception as e:
    print(f"Warning: Could not initialize SemanticSearcher: {e}")
    searcher = None

llm = init_chat_model("gpt-4.1-mini", model_provider="openai")

@tool(response_format="content_and_artifact")
def search_technical_docs(query: str):
    """Search for technical information from Dexa Medica FAQ documents."""
    try:
        if searcher is None:
            return "Technical search service is not available.", []
            
        results = searcher.search(query, limit=5)
        
        if not results:
            return "No relevant technical information found.", []
        
        # Format the results
        formatted_results = []
        for result in results:
            formatted_result = f"Category: {result['category']}\n"
            if result['question']:
                formatted_result += f"Question: {result['question']}\n"
            if result['answer']:
                formatted_result += f"Answer: {result['answer']}\n"
            formatted_result += f"Similarity: {result['similarity']:.3f}\n"
            formatted_results.append(formatted_result)
        
        serialized = "\n\n".join(formatted_results)
        return serialized, results
    except Exception as e:
        return f"Error searching technical documents: {str(e)}", []

def query_or_respond(state: MessagesState):
    """Generate tool call for technical search or respond."""
    llm_with_tools = llm.bind_tools([search_technical_docs])
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

tools = ToolNode([search_technical_docs])

def generate(state: MessagesState):
    """Generate response based on retrieved technical information."""
    system_message = SystemMessage(content="""You are a technical assistant specializing in Dexa Medica FAQ. 
    Use the retrieved technical information to provide accurate, helpful answers about Dexa Medica products, 
    services, and technical questions. If the information is not sufficient, say so clearly.""")
    
    response = llm.invoke([system_message] + state["messages"])
    return {"messages": [response]}

# Create the graph
graph = (
    StateGraph(MessagesState)
    .add_node(query_or_respond)
    .add_node(tools)
    .add_node(generate)
    .set_entry_point("query_or_respond")
    .add_conditional_edges(
        "query_or_respond",
        tools_condition,
        {END: END, "tools": "tools"},
    )
    .add_edge("tools", "generate")
    .add_edge("generate", END)
    .compile(name="Technical")
) 