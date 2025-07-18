import streamlit as st
from typing import Dict, List, Any, cast, TypedDict
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from dataclasses import dataclass
import json
import os
from langsmith.run_helpers import traceable
from rouge_score import rouge_scorer

__all__ = ['graph', 'evaluate_agent', 'AgentState']

# Import vector search functionality
from vector.search import SemanticSearcher

# Initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

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
    ground_truth: str

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

def evaluate_response(generated_answer: str, ground_truth: str) -> Dict[str, float]:
    """
    Evaluate the generated answer against ground truth using ROUGE scores
    """
    scores = scorer.score(ground_truth, generated_answer)
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }

# Define workflow nodes
@traceable(name="process_query")
def process_query(state: AgentState) -> AgentState:
    """Process the initial user query"""
    messages = state["messages"]
    last_message = messages[-1]
    
    if isinstance(last_message, HumanMessage):
        query = str(last_message.content)
        # Initialize or reset state
        state.update({
            "original_query": query,
            "search_attempts": 0,
            "answer_found": False,
            "modified_queries": [],
            "search_results": [],
            "max_attempts": 3,
            "similarity_threshold": 1.5  # Updated to match our distance-based threshold
        })
        
        # Debug print
        print("Initialized state with query:", query)
    
    return state

@traceable(name="perform_search")
def perform_search(state: AgentState) -> AgentState:
    """Perform similarity search on the vector database"""
    query = state["original_query"]
    
    # Use modified query if available
    if state["modified_queries"]:
        query = state["modified_queries"][-1]
    
    try:
        # Perform search and store results directly in state
        results = searcher.search(query, limit=5)
        state["search_results"] = results
        state["search_attempts"] += 1
        
        # Debug print
        print(f"Search attempt {state['search_attempts']}")
        print(f"Query: {query}")
        print(f"Found {len(results)} results")
        print(f"Best similarity score: {results[0]['similarity'] if results else 'N/A'}")
        
        # Add search results to messages for model context
        search_context = f"Search results for query: '{query}'\n\n"
        for i, result in enumerate(results, 1):
            search_context += f"Result {i}:\n"
            search_context += f"Question: {result['question']}\n"
            search_context += f"Answer: {result['answer']}\n"
            search_context += f"Similarity: {result['similarity']:.2f}\n\n"
        
        state["messages"].append(SystemMessage(content=search_context))
        
    except Exception as e:
        error_msg = f"Error during search: {str(e)}"
        print(error_msg)  # Debug print
        state["messages"].append(SystemMessage(content=error_msg))
        state["search_results"] = []
    
    return state

@traceable(name="check_answer_quality")
def check_answer_quality(state: AgentState) -> AgentState:
    """Check if the search results contain a good answer"""
    results = state["search_results"]
    
    # Debug print
    print(f"Checking {len(results)} results for quality")
    
    if not results:
        print("No results found")
        state["answer_found"] = False
        return state
    
    # Check if any result has good similarity (low distance) and contains an answer
    for result in results:
        similarity = result["similarity"]
        has_answer = bool(result.get("answer", "").strip())
        print(f"Checking result - Similarity: {similarity}, Has answer: {has_answer}")
        
        if similarity <= state["similarity_threshold"] and has_answer:
            print(f"Found good answer with similarity {similarity}")
            state["answer_found"] = True
            break
    
    return state

@traceable(name="generate_answer")
def generate_answer(state: AgentState) -> AgentState:
    """Generate final answer based on search results"""
    messages = state["messages"]
    original_query = state["original_query"]
    results = state["search_results"]
    
    if not results:
        response_content = "Maaf, saya tidak dapat menemukan informasi yang relevan untuk pertanyaan Anda. Silakan coba pertanyaan yang berbeda atau lebih spesifik."
    else:
        # Find the best matching result
        best_result = results[0]  # Results are already sorted by similarity
        
        # Clean and format the answer
        answer = best_result['answer']
        # Replace 'o' bullet points with proper dashes and fix formatting
        answer = answer.replace('\no  ', '\n- ')  # Replace 'o  ' bullet points
        answer = answer.replace('\no', '\n- ')    # Replace 'o' bullet points
        answer = answer.replace('\n\n', '\n')    # Remove double newlines
        # Clean up empty lines but preserve structure
        lines = answer.split('\n')
        cleaned_lines = []
        for line in lines:
            if line.strip():  # Non-empty line
                cleaned_lines.append(line)
            elif cleaned_lines and not cleaned_lines[-1].strip():  # Don't add multiple empty lines
                continue
            else:
                cleaned_lines.append(line)
        answer = '\n'.join(cleaned_lines).strip()

        print(f"Answer: {answer}")
        print(f"Best result: {best_result}")
        
        # Create context focusing on the best result
        context = f"""
        Pertanyaan yang ditemukan:
        {best_result['question']}
        
        Jawaban yang tepat:
        {answer}
        """
        
        # Add supporting information from other highly relevant results
        supporting_info = []
        for result in results[1:]:
            if result['similarity'] > 0.7:
                clean_answer = result['answer'].replace('\n', ' ').strip()
                if clean_answer:
                    supporting_info.append(clean_answer)
        
        if supporting_info:
            context += "\n\nInformasi tambahan terkait:\n"
            for info in supporting_info:
                context += f"- {info}\n"

        # Generate response using the model
        system_prompt = f"""
        Anda adalah asisten FAQ Dexa Medica. Tugas Anda adalah memberikan jawaban yang tepat dan terformat dengan baik.
        
        Pertanyaan pengguna: {original_query}
        
        Informasi tersedia:
        {context}
        
        Instruksi penting:
        1. Gunakan jawaban yang tepat dari database secara langsung
        2. Pertahankan format poin-poin bila ada dalam jawaban
        3. Jangan tambahkan kata-kata pengantar atau penutup
        4. Jangan ubah atau tambahkan informasi
        5. Jangan gunakan frasa seperti "berdasarkan informasi" atau "menurut data"
        6. Berikan jawaban langsung dan to the point
        """
        
        response_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=original_query)
        ]
        
        try:
            response = model.invoke(response_messages)
            response_content = str(response.content)
            
            # If ground truth is available, evaluate the response
            if "ground_truth" in state and state["ground_truth"]:
                print(f"Evaluating response against ground truth: {state['ground_truth']}")
                evaluate_response(response_content, state["ground_truth"])
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            response_content = f"Terjadi kesalahan saat memproses jawaban: {str(e)}"
    
    # Add AI response to messages
    state["messages"].append(AIMessage(content=response_content))
    return state

@traceable(name="modify_query")
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
    # Stop if we found a good answer
    if state["answer_found"]:
        return "generate_answer"
    
    # Stop if we've reached max attempts
    if state["search_attempts"] >= state["max_attempts"]:
        return "generate_answer"
    
    # Stop if we have results with good similarity (low distance)
    # Note: Our scores are distances, so LOWER is better
    if state["search_results"] and any(result["similarity"] <= 1.5 for result in state["search_results"]):
        return "generate_answer"
    
    # Continue searching only if we have no good results and haven't hit max attempts
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

# Function to run evaluation
def evaluate_agent(query: str, ground_truth: str) -> Dict[str, Any]:
    """
    Run the agent with a query and evaluate its response against ground truth
    """
    # Initialize state with ground truth
    state = AgentState(
        messages=[HumanMessage(content=query)],
        search_results=[],
        original_query="",
        modified_queries=[],
        search_attempts=0,
        max_attempts=3,
        similarity_threshold=0.7,
        answer_found=False,
        ground_truth=ground_truth
    )
    
    # Run the graph
    final_state = graph.invoke(state)
    
    # Get the generated answer
    generated_answer = str(final_state["messages"][-1].content)
    
    # Calculate ROUGE scores
    scores = evaluate_response(generated_answer, ground_truth)
    
    return {
        "query": query,
        "generated_answer": generated_answer,
        "ground_truth": ground_truth,
        "rouge_scores": scores,
        "search_attempts": final_state["search_attempts"],
        "answer_found": final_state["answer_found"]
    } 