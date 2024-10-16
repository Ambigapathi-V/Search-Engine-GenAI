import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

# Page configuration for better UX
st.set_page_config(page_title="Chatbot with Search", page_icon="💬", layout="centered")

# Custom CSS for improved UI
st.markdown("""
    <style>
        .assistant { background-color: #f1f8e9; border-radius: 10px; padding: 10px; margin: 10px 0;}
        .user { background-color: #e3f2fd; border-radius: 10px; padding: 10px; margin: 10px 0;}
        .message-container { max-width: 700px; margin: auto; }
        .sidebar .sidebar-content { font-size: 16px; }
        .title { text-align: center; font-weight: bold; color: #2E8B57; }
        .input-box { margin-top: 20px; }
    </style>
""", unsafe_allow_html=True)

# Initialize Arxiv and Wikipedia Tools
api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

search = DuckDuckGoSearchRun(name='Search')

# App title
st.markdown("<h1 class='title'>Data Digger: Chat with Search</h1>", unsafe_allow_html=True)

# Sidebar for settings
with st.sidebar:
    st.header("Settings ⚙️")
    st.write("Please enter your Groq API key to begin.")
    api_key = st.text_input('Groq API Key', type='password', placeholder="Enter your API key")
    st.markdown("---")

# Initialize session state for messages
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {'role': 'assistant', 'content': 'Hi, I am a chatbot who can search the web. How can I help you?'}
    ]

# Display chat messages in a more visually distinct way
st.markdown("<div class='message-container'>", unsafe_allow_html=True)

for msg in st.session_state['messages']:
    role_class = "assistant" if msg['role'] == 'assistant' else "user"
    st.markdown(f"<div class='{role_class}'>{msg['content']}</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# Handle user input and chatbot response
if prompt := st.chat_input(placeholder='Type your question...'):
    # Append user message to session state
    st.session_state['messages'].append({'role': 'user', 'content': prompt})
    st.markdown(f"<div class='user'>{prompt}</div>", unsafe_allow_html=True)

    # Show loading spinner while processing
    with st.spinner('Assistant is thinking...'):
        # Load environment variables
        load_dotenv()

        # Initialize GroQ Agent
        llm = ChatGroq(
            api_key=api_key,
            model_name='gemma-7b-it',
            streaming=True
        )

        # Initialize LangChain Agent with tools
        tools = [search, arxiv, wiki]
        agent = initialize_agent(
            tools,
            llm,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True
        )

        # Get assistant response using Streamlit callback handler
        with st.chat_message('assistant'):
            st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            try:
                response = agent.run(prompt, callbacks=[st_cb])
                # Append assistant response to session state
                st.session_state['messages'].append({'role': 'assistant', 'content': response})
                st.markdown(f"<div class='assistant'>{response}</div>", unsafe_allow_html=True)
            except Exception as e:
                # Handle errors and inform the user
                st.session_state['messages'].append({'role': 'assistant', 'content': "I encountered an issue. Please try again!"})
                st.markdown(f"<div class='assistant'>I encountered an issue. Please try again!</div>", unsafe_allow_html=True)
