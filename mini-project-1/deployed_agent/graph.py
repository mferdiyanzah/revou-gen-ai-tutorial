import streamlit as st
from typing import Dict, List, Any, cast
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from dataclasses import dataclass
import json

# Import vector search functionality
from vector.search import SemanticSearcher

# Initialize chat model
model = init_chat_model("gpt-4.1-nano", model_provider="openai")

# Define custom state for our agent
class AgentState(MessagesState):
    """Extended state to include search results and context"""
    search_results: List[Dict[str, Any]]
    original_query: str
    modified_queries: List[str]
    search_attempts: int
    max_attempts: int
    similarity_threshold: float
    answer_found: bool

# Initialize components
searcher = SemanticSearcher()
memory = MemorySaver()

# Define the similarity search tool
@tool
def similarity_search(query: str, limit: int = 5) -> str:
    """
    Perform similarity search on the FAQ database.
    """
    try:
        results = searcher.search(query, limit)
        return json.dumps(results, indent=2)
    except Exception as e:
        return f"Error performing search: {str(e)}"

# Create tool node
tools = [similarity_search]
tool_node = ToolNode(tools)

# Define workflow nodes
def process_query(state: AgentState) -> AgentState:
    """Process the initial user query"""
    messages = state["messages"]
    last_message = messages[-1]
    
    if isinstance(last_message, HumanMessage):
        query = str(last_message.content)
        state["original_query"] = query
        state["search_attempts"] = 0
        state["answer_found"] = False
        state["modified_queries"] = []
        state["search_results"] = []
        state["max_attempts"] = 3  # Maximum number of search attempts
        state["similarity_threshold"] = 0.7  # Minimum similarity score for a good match
    
    return state

def perform_search(state: AgentState) -> AgentState:
    """Perform similarity search on the vector database"""
    query = state["original_query"]
    
    # Use modified query if available
    if state["modified_queries"]:
        query = state["modified_queries"][-1]
    
    try:
        # Perform search
        results = searcher.search(query, limit=5)
        state["search_results"] = results
        state["search_attempts"] += 1
        
        # Add search results to messages for model context
        search_context = f"Search results for query: '{query}'\n\n"
        for i, result in enumerate(results, 1):
            search_context += f"Result {i}:\n"
            search_context += f"Category: {result['category']}\n"
            search_context += f"Question: {result['question']}\n"
            search_context += f"Answer: {result['answer']}\n"
            search_context += f"Similarity: {result['similarity']:.2%}\n\n"
        
        state["messages"].append(SystemMessage(content=search_context))
        
    except Exception as e:
        error_msg = f"Error during search: {str(e)}"
        state["messages"].append(SystemMessage(content=error_msg))
        state["search_results"] = []
    
    return state

def check_answer_quality(state: AgentState) -> AgentState:
    """Check if the search results contain a good answer"""
    results = state["search_results"]
    
    if not results:
        state["answer_found"] = False
        return state
    
    # Check if any result has high similarity and contains an answer
    for result in results:
        if (result["similarity"] >= state["similarity_threshold"] and 
            result["answer"] and 
            len(result["answer"].strip()) > 10):
            state["answer_found"] = True
            break
    
    return state

def generate_answer(state: AgentState) -> AgentState:
    """Generate final answer based on search results"""
    messages = state["messages"]
    original_query = state["original_query"]
    results = state["search_results"]
    
    if not results:
        response_content = "Maaf, saya tidak dapat menemukan informasi yang relevan untuk pertanyaan Anda. Silakan coba pertanyaan yang berbeda atau lebih spesifik."
    else:
        # Create context from search results
        context = "Berdasarkan informasi yang tersedia:\n\n"
        for i, result in enumerate(results[:3], 1):  # Use top 3 results
            if result["answer"]:
                context += f"{i}. {result['question']}\n"
                context += f"   Jawaban: {result['answer']}\n\n"
        # Generate response using the model
        system_prompt = f"""
        Anda adalah asisten FAQ untuk Dexa Medica. Gunakan informasi berikut untuk menjawab pertanyaan pengguna dengan akurat dan membantu.
        
        Konteks informasi:
        {context}
        
        Pertanyaan pengguna: {original_query}
        
        Instruksi:
        1. Berikan jawaban yang akurat berdasarkan informasi yang tersedia
        2. Jika informasi tidak lengkap, jelaskan apa yang bisa dijawab
        3. Gunakan bahasa Indonesia yang ramah dan profesional
        4. Jika tidak ada informasi yang relevan, sarankan untuk menghubungi customer service
        """
        response_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=original_query)
        ]
        try:
            response = model.invoke(response_messages)
            response_content = response.content
        except Exception as e:
            response_content = f"Terjadi kesalahan saat memproses jawaban: {str(e)}"
    # Add AI response to messages
    state["messages"].append(AIMessage(content=response_content))
    return state

def modify_query(state: AgentState) -> AgentState:
    """Modify the query for better search results"""
    original_query = state["original_query"]
    attempts = state["search_attempts"]
    
    # Generate modified query using the model
    modification_prompt = f"""
    Query asli: "{original_query}"
    Percobaan ke: {attempts + 1}
    
    Hasilkan query pencarian yang dimodifikasi untuk meningkatkan hasil pencarian FAQ Dexa Medica.
    Pertimbangkan:
    1. Sinonim dan kata kunci alternatif
    2. Konteks bisnis farmasi/kesehatan
    3. Penyederhanaan atau perluasan query
    
    Berikan hanya query yang dimodifikasi tanpa penjelasan:
    """
    
    try:
        modification_messages = [
            SystemMessage(content=modification_prompt),
            HumanMessage(content=original_query)
        ]
        
        response = model.invoke(modification_messages)
        modified_query = str(response.content).strip()
        
        # Add to modified queries list
        state["modified_queries"].append(modified_query)
        
    except Exception as e:
        # Fallback modification strategies
        if attempts == 1:
            modified_query = f"informasi tentang {original_query}"
        elif attempts == 2:
            modified_query = f"FAQ {original_query}"
        else:
            modified_query = original_query
        
        state["modified_queries"].append(modified_query)
    
    return state

def should_retry_search(state: AgentState) -> str:
    """Determine if we should retry the search with a modified query"""
    if state["answer_found"]:
        return "generate_answer"
    elif state["search_attempts"] >= state["max_attempts"]:
        return "generate_answer"
    else:
        return "modify_query"

# Build the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("process_query", process_query)
workflow.add_node("perform_search", perform_search)
workflow.add_node("check_answer_quality", check_answer_quality)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("modify_query", modify_query)

# Add edges
workflow.add_edge(START, "process_query")
workflow.add_edge("process_query", "perform_search")
workflow.add_edge("perform_search", "check_answer_quality")
workflow.add_conditional_edges(
    "check_answer_quality",
    should_retry_search,
    {
        "generate_answer": "generate_answer",
        "modify_query": "modify_query"
    }
)
workflow.add_edge("modify_query", "perform_search")
workflow.add_edge("generate_answer", END)

# Compile with memory and expose
graph = workflow.compile(name="faq_agent") 