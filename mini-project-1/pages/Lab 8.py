import streamlit as st
import os
from langchain_core.messages import AIMessageChunk, HumanMessage, AIMessage, SystemMessage
import agents.graph as gr
import agents.DBQNA as DBQNA
import agents.RAG as RAG
import agents.Technical as Technical
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.types import Command
from typing import Literal
from pydantic import BaseModel, Field
from langgraph.checkpoint.memory import InMemorySaver

st.title("Simple Graph with Streamlit")

from dotenv import load_dotenv
load_dotenv(override=True)

def get_stream():
    for chunk, metadata in gr.agent.stream({"messages":"what is 4 + 7"}, stream_mode="messages"):
        if isinstance(chunk, AIMessageChunk):
            yield chunk

st.write_stream(get_stream)

# Set default DB_PATH if not provided in environment
DB_PATH = os.environ.get('DB_PATH', '../sqlite/chinook.db')

from langchain.chat_models import init_chat_model
model = init_chat_model("gpt-4.1-mini", model_provider= "openai")

class BestAgent(BaseModel):
    agent_name: str = Field(description = "The best agent to handle specific request from users.")

class SupervisorState(MessagesState):
    user_question: str 

def supervisor(state: SupervisorState) -> Command[Literal["DBQNA", "RAG", "Technical"]]:
    last_message = state["messages"][-1]
    instruction = [SystemMessage(content=f"""You receive the following question from users. Decide which agent is the most suitable for completing the task.
                                    Delegate to DBQNA agent if users ask a question that can be answered by data inside a database. 
                                    Delegate to RAG agent if users ask a question about Dexa Medica company information or general information.
                                    Delegate to Technical agent if users ask technical questions about Dexa Medica FAQ, products, or services.
                                 """)]
    model_with_structure = model.with_structured_output(BestAgent)
    response = model_with_structure.invoke(instruction + [last_message])
    return Command(
        update= {'user_question': last_message.content},
        goto=getattr(response, 'agent_name', 'RAG')
    )

def callRAG(state: SupervisorState):
    prompt = state['user_question']
    response = RAG.graph.invoke({"messages":HumanMessage(content=prompt)})
    return {"messages": [response['messages'][-1]]}

def callDBQNA(state: SupervisorState):
    prompt = state['user_question']
    response = DBQNA.graph.invoke({"messages":HumanMessage(content=prompt), "db_name": DB_PATH, "user_question" : prompt})
    return {"messages": [response['messages'][-1]]}

def callTechnical(state: SupervisorState):
    prompt = state['user_question']
    response = Technical.graph.invoke({"messages":HumanMessage(content=prompt)})
    return {"messages": [response['messages'][-1]]}

# memory = InMemorySaver()
supervisor_agent = (
    StateGraph(SupervisorState)
    .add_node(supervisor)
    .add_node("RAG", callRAG)
    .add_node("DBQNA", callDBQNA)
    .add_node("Technical", callTechnical)
    .add_edge(START, "supervisor")
    .add_edge("RAG", END)
    .add_edge("DBQNA", END)
    .add_edge("Technical", END)
    .compile(name= "supervisor")
)

prompt = st.chat_input("Write your question here ... ")
if prompt:
    with st.chat_message("human"):
        st.markdown(prompt)

    final_answer = ""
    with st.chat_message("ai"):
        status_placeholder = st.empty()
        answer_placeholder = st.empty()
        status_placeholder.status(label="Process Start")
        state = "Process Start"
        for chunk, metadata in supervisor_agent.stream({"messages":HumanMessage(content=prompt)}, stream_mode="messages"):
            # Access metadata as a dictionary
            node_name = metadata.get('langgraph_node', '') if isinstance(metadata, dict) else ''
            if node_name != state:
                status_placeholder.status(label=node_name)
                state = node_name
                final_answer = "" 
            
            if node_name == "final_answer":
                if hasattr(chunk, 'content'):
                    final_answer += str(getattr(chunk, 'content', ''))
                    answer_placeholder.markdown(final_answer)
            
            if node_name == "generate":
                if hasattr(chunk, 'content'):
                    final_answer += str(getattr(chunk, 'content', ''))
                    answer_placeholder.markdown(final_answer)
        
        status_placeholder.status(label="Complete", state='complete')

# DBQNA.graph.stream({"messages":HumanMessage(content=prompt), "db_name": DB_PATH, "user_question" : prompt}, stream_mode="messages")
# RAG.graph.stream({"messages":HumanMessage(content=prompt)}, stream_mode="messages")
# Technical.graph.stream({"messages":HumanMessage(content=prompt)}, stream_mode="messages")
            