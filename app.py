import streamlit as st
import os
import tempfile
from operator import add as add_messages
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, BaseMessage
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Sequence
from langchain_ollama import OllamaEmbeddings
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS # Using FAISS for faster in-memory storage

# --- 1. Caching: Load Models Once ---
# We use @st.cache_resource to load the heavy LLM and embedding models only once.

# docker changes
@st.cache_resource
def get_llm():
    """Loads the Qwen3:8b model via Ollama."""
    print("Loading LLM model...")
    # This tells the app to find Ollama at the 'ollama' service address
    return ChatOllama(model="qwen3:8b", base_url="http://ollama:11434")


@st.cache_resource
def get_embeddings():
    """Loads the Ollama embedding model."""
    print("Loading embedding model...")
    # This tells the app to find Ollama at the 'ollama' service address
    return OllamaEmbeddings(model="qwen3-embedding:4b", base_url="http://ollama:11434")

# --- 2. LangGraph Agent Definition ---
# All agent-related code is defined here.

# Define the state for our graph
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# System prompt for the agent
system_prompt = """
You are an intelligent AI assistant who answers questions about the PDF document loaded into your knowledge base.
Use the retriever_tool available to answer questions. You can make multiple calls if needed.
If you need to look up some information before asking a follow-up question, you are allowed to do that.
Please always cite the specific parts of the documents you use in your answers.
"""

def should_continue(state: AgentState):
    """Check if the last message contains tool calls."""
    result = state['messages'][-1]
    return hasattr(result, 'tool_calls') and len(result.tool_calls) > 0

def call_llm(state: AgentState):
    """Function to call the LLM with the current state."""
    messages = list(state['messages'])
    
    # Add system prompt if not present
    if not isinstance(messages[0], SystemMessage):
         messages = [SystemMessage(content=system_prompt)] + messages
    
    # --- FIX: Get the TOOL-BOUND LLM from session state ---
    if "llm_with_tools" not in st.session_state:
        # This is a fallback, but should not be hit if UI is correct
        return {'messages': [HumanMessage(content="Error: LLM not initialized. Please (re)upload your PDF.")]}
        
    llm = st.session_state.llm_with_tools # Get the tool-bound LLM
    # --- END FIX ---

    message = llm.invoke(messages)
    return {'messages': [message]}
         
    llm = get_llm() # Get the cached LLM
    message = llm.invoke(messages)
    return {'messages': [message]}

def take_action(state: AgentState):
    """Execute tool calls from the LLM's response."""
    tool_calls = state['messages'][-1].tool_calls
    results = []
    
    # The retriever_tool is stored in st.session_state
    if "retriever_tool" not in st.session_state:
        return {'messages': [HumanMessage(content="Error: Retriever tool not initialized. Please upload a PDF.")]}

    retriever_tool = st.session_state.retriever_tool
    tools_dict = {retriever_tool.name: retriever_tool}

    for t in tool_calls:
        print(f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}")
        
        if not t['name'] in tools_dict:
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry."
        else:
            result = tools_dict[t['name']].invoke(t['args'].get('query', ''))
        
        results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))

    print("Tools Execution Complete. Back to the model!")
    return {'messages': results}

# --- 3. Streamlit App UI ---

st.set_page_config(page_title="RAG Agent: Chat with Your PDF", layout="wide")
st.title("📄 RAG Agent: Chat with Your PDF (qwen3:8b)")

# --- 4. PDF Upload and Processing ---
with st.sidebar:
    st.header("1. Upload Your PDF")
    uploaded_file = st.file_uploader("Upload a PDF file to chat with", type="pdf")

    if uploaded_file:
        # Check if this is a new file
        if st.session_state.get("uploaded_file_name") != uploaded_file.name:
            with st.spinner(f"Processing '{uploaded_file.name}'... This may take a moment."):
                
                # 1. Save temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                # 2. Load PDF
                try:
                    loader = PyPDFLoader(tmp_file_path)
                    pages = loader.load()
                    
                    # 3. Split Text
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    pages_split = text_splitter.split_documents(pages)

                    # 4. Create Vector Store (using cached embedding model)
                    embeddings = get_embeddings()
                    vectorstore = FAISS.from_documents(documents=pages_split, embedding=embeddings)

                    # 5. Create Retriever
                    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

                    # 6. Create the dynamic retriever tool
                    @tool
                    def retriever_tool(query: str) -> str:
                        """
                        This tool searches and returns information from the uploaded PDF document.
                        """
                        docs = retriever.invoke(query)
                        if not docs:
                            return "I found no relevant information in the document."
                        results = []
                        for i, doc in enumerate(docs):
                            results.append(f"Document {i+1}:\n{doc.page_content}")
                        return "\n\n".join(results)

                    # 7. Bind tool to the cached LLM
                    llm_with_tools = get_llm().bind_tools([retriever_tool])# Store in session state 
                    st.session_state.llm_with_tools = llm_with_tools
                    

                    # 8. Create and compile the graph
                    graph = StateGraph(AgentState)
                    graph.add_node("llm", call_llm)
                    graph.add_node("retriever_agent", take_action)
                    graph.add_conditional_edges("llm", should_continue, {True: "retriever_agent", False: END})
                    graph.add_edge("retriever_agent", "llm")
                    graph.set_entry_point("llm")
                    rag_agent = graph.compile()
                    
                    # 9. Store agent, tool, and history in session state
                    st.session_state.rag_agent = rag_agent
                    st.session_state.retriever_tool = retriever_tool # Store the tool
                    st.session_state.messages = [] # Reset chat history
                    st.session_state.uploaded_file_name = uploaded_file.name
                    
                    st.success(f"PDF '{uploaded_file.name}' processed! Ready to chat.")

                except Exception as e:
                    st.error(f"Error loading PDF: {e}")
                
                # Clean up the temporary file
                finally:
                    if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                        os.remove(tmp_file_path)

# Initialize messages in session state if not present
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 5. Chat Interface ---

# Display chat history
for message in st.session_state.messages:
    # Check if message is a BaseMessage object or a dict (from older runs)
    if isinstance(message, BaseMessage):
        role = message.type
        content = message.content
    elif isinstance(message, dict):
        role = message.get("role", "user") # Default to user for safety
        content = message.get("content", "")
    
    # Don't display tool messages or system messages
    if role not in ["tool", "system"]:
        with st.chat_message(role):
            st.markdown(content)

# Get user input
if prompt := st.chat_input("Ask a question about your PDF..."):
    # Ensure a PDF has been processed
    if "rag_agent" not in st.session_state:
        st.error("Please upload a PDF file in the sidebar first.")
    else:
        # Add user message to state and display it
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get the agent from session state
        rag_agent = st.session_state.rag_agent
        
        # Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Invoke the agent with the *entire* message history
                try:
                    result = rag_agent.invoke({"messages": st.session_state.messages})
                    
                    # The final message is the assistant's response
                    response = result['messages'][-1]
                    st.markdown(response.content)
                    
                    # Add all new messages (including tool calls) to history
                    st.session_state.messages.extend(result['messages'][len(st.session_state.messages):])

                except Exception as e:
                    st.error(f"An error occurred: {e}")