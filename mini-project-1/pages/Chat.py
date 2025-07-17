import streamlit as st
import json
from datetime import datetime
import uuid

# Configure page
st.set_page_config(
    page_title="Chat with Dexa Medica",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {}
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Custom CSS for ChatGPT-like styling
st.markdown("""
<style>
    /* Hide streamlit style */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #202123;
    }
    
    /* Chat container */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
    }
    
    /* Message styling */
    .user-message {
        background-color: #f7f7f8;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        margin-left: 50px;
    }
    
    .assistant-message {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        margin-right: 50px;
        border: 1px solid #e5e5e5;
    }
    
    /* Input styling */
    .stTextInput > div > div > input {
        border-radius: 20px;
        border: 2px solid #e5e5e5;
        padding: 12px 20px;
    }
    
    /* Sidebar button styling */
    .sidebar-button {
        width: 100%;
        margin: 5px 0;
        padding: 10px;
        border: none;
        border-radius: 8px;
        background-color: #343541;
        color: white;
        text-align: left;
        cursor: pointer;
    }
    
    .sidebar-button:hover {
        background-color: #40414f;
    }
    
    .active-chat {
        background-color: #40414f;
    }
    
    /* New chat button */
    .new-chat-btn {
        background-color: #10a37f;
        color: white;
        border: none;
        padding: 12px 20px;
        border-radius: 8px;
        width: 100%;
        margin-bottom: 20px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def create_new_chat():
    """Create a new chat session"""
    session_id = str(uuid.uuid4())
    st.session_state.chat_sessions[session_id] = {
        "title": "New Chat",
        "messages": [],
        "created_at": datetime.now().isoformat()
    }
    st.session_state.current_session_id = session_id
    st.session_state.messages = []
    st.rerun()

def load_chat_session(session_id):
    """Load a specific chat session"""
    if session_id in st.session_state.chat_sessions:
        st.session_state.current_session_id = session_id
        st.session_state.messages = st.session_state.chat_sessions[session_id]["messages"]
        st.rerun()

def save_current_session():
    """Save current session to chat_sessions"""
    if st.session_state.current_session_id:
        st.session_state.chat_sessions[st.session_state.current_session_id]["messages"] = st.session_state.messages
        # Update title based on first user message
        if st.session_state.messages and st.session_state.chat_sessions[st.session_state.current_session_id]["title"] == "New Chat":
            first_user_msg = next((msg["content"] for msg in st.session_state.messages if msg["role"] == "user"), "New Chat")
            st.session_state.chat_sessions[st.session_state.current_session_id]["title"] = first_user_msg[:30] + "..." if len(first_user_msg) > 30 else first_user_msg

def delete_chat_session(session_id):
    """Delete a chat session"""
    if session_id in st.session_state.chat_sessions:
        del st.session_state.chat_sessions[session_id]
        if st.session_state.current_session_id == session_id:
            st.session_state.current_session_id = None
            st.session_state.messages = []
        st.rerun()

# Sidebar - Chat History
with st.sidebar:
    st.markdown("# 💬 Chat History")
    
    # New Chat Button
    if st.button("+ New Chat", key="new_chat", help="Start a new conversation"):
        create_new_chat()
    
    st.markdown("---")
    
    # Display chat sessions
    if st.session_state.chat_sessions:
        for session_id, session_data in reversed(list(st.session_state.chat_sessions.items())):
            col1, col2 = st.columns([4, 1])
            
            with col1:
                # Chat session button
                is_active = session_id == st.session_state.current_session_id
                button_style = "active-chat" if is_active else ""
                
                if st.button(
                    session_data["title"], 
                    key=f"chat_{session_id}",
                    help=f"Created: {datetime.fromisoformat(session_data['created_at']).strftime('%Y-%m-%d %H:%M')}"
                ):
                    load_chat_session(session_id)
            
            with col2:
                # Delete button
                if st.button("🗑️", key=f"delete_{session_id}", help="Delete this chat"):
                    delete_chat_session(session_id)
    else:
        st.markdown("*No chat history yet*")
        st.markdown("Start a new conversation!")

# Main Content Area
st.title("🏥 Chat with Dexa Medica")
st.markdown("Ask me anything about Dexa Medica's products, services, and FAQ!")

# Create initial session if none exists
if not st.session_state.current_session_id and not st.session_state.chat_sessions:
    create_new_chat()

# Display chat messages
chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask about Dexa Medica..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate assistant response (placeholder for now)
    with st.chat_message("assistant"):
        response = f"Thank you for asking about '{prompt}'. I'm currently being configured to answer questions about Dexa Medica. This is a placeholder response that will be replaced with actual AI-powered responses soon!"
        st.markdown(response)
    
    # Add assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Save session
    save_current_session()
    
    # Rerun to update the interface
    st.rerun()

# Footer with tips
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9em;'>
💡 <strong>Tips:</strong> Ask about Dexa Medica products, pricing, availability, or general FAQ questions<br>
🔄 Use the sidebar to manage your chat history and start new conversations
</div>
""", unsafe_allow_html=True)