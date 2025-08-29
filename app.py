import streamlit as st
import asyncio
import datetime
import logging
import sys
import os

logging.basicConfig(
    level=logging.INFO,
    format='🎯 %(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logging.getLogger('streamlit').setLevel(logging.WARNING)
logging.getLogger('tornado').setLevel(logging.WARNING) 
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('blinker').setLevel(logging.WARNING)
logging.getLogger('watchdog').setLevel(logging.WARNING)
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
logging.getLogger('ibm_watsonx_ai').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core import create_simple_rag_engine
from agent import create_agent


def run_async(coro):
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import nest_asyncio
            nest_asyncio.apply()
            return asyncio.run(coro)
        else:
            return asyncio.run(coro)
    except RuntimeError:
        try:
            import nest_asyncio
            nest_asyncio.apply()
            return asyncio.run(coro)
        except ImportError:
            st.error("Install nest_asyncio: pip install nest_asyncio")
            return None

async def process_pdfs(uploaded_files, model_id):
    try:
        logger.info(f"Starting PDF processing with model: {model_id}")
        logger.info(f"Processing {len(uploaded_files)} files")
        
        engine = await create_simple_rag_engine(uploaded_files)
        logger.info("RAG engine created successfully")
        
        agent = await create_agent(engine, model_id)
        logger.info(f"Agent created successfully with model: {model_id}")
        
        return agent
    except Exception as e:
        logger.error(f"Error in process_pdfs: {e}", exc_info=True)
        st.error(f"Error: {e}")
        return None

async def answer_question(agent, question):
    try:
        logger.info(f"Answering question: {question[:50]}...")
        result = await agent.query_async(question)
        logger.info(f"Question answered successfully: {result.get('status', 'unknown')}")
        
        if result.get("success", False):
            return result.get("answer", ""), result.get("sources", [])
        else:
            return result.get("answer", f"Error: {result.get('error', 'Unknown error')}"), []
    except Exception as e:
        logger.error(f"Error in answer_question: {e}", exc_info=True)
        return f"Error: {e}", []

st.set_page_config(page_title="StudyMate", layout="wide")
st.title("📚 StudyMate")

SUPPORTED_MODELS = [
    'mistralai/mistral-large',
]

if "history" not in st.session_state:
    st.session_state.history = []
if "agent" not in st.session_state:
    st.session_state.agent = None
if "selected_model" not in st.session_state:
    st.session_state.selected_model = SUPPORTED_MODELS[0]

if st.session_state.selected_model not in SUPPORTED_MODELS:
    st.session_state.selected_model = SUPPORTED_MODELS[0]

with st.sidebar:
    st.header("🤖 Model Selection")
    selected_model = st.selectbox(
        "Choose AI Model:",
        SUPPORTED_MODELS,
        index=SUPPORTED_MODELS.index(st.session_state.selected_model),
        help="Select the Watson AI model to use for answering questions"
    )
    st.session_state.selected_model = selected_model
    
    st.header("📄 Upload PDFs")
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    
    if uploaded_files and st.button("Process Documents"):
        st.session_state.agent = None
        st.session_state.history = []
        
        with st.spinner(f"Processing with {selected_model}..."):
            agent = run_async(process_pdfs(uploaded_files, selected_model))
            if agent:
                st.session_state.agent = agent
                st.success(f"✅ Processed {len(uploaded_files)} files with {selected_model}")
                st.balloons()

if st.session_state.agent is None:
    st.info("📋 Upload and process PDFs to begin")
    st.markdown("### How to get started:")
    st.markdown("1. 🤖 Choose your preferred AI model from the sidebar")
    st.markdown("2. 📄 Upload one or more PDF documents")
    st.markdown("3. 🔄 Click 'Process Documents' to analyze them")
    st.markdown("4. ❓ Ask questions about your documents!")
else:
    st.success(f"🤖 Currently using: **{st.session_state.selected_model}**")
    question = st.text_input("Ask your question:")
    
    if question:
        with st.spinner("Generating answer..."):
            answer, sources = run_async(answer_question(st.session_state.agent, question))
            
            st.subheader("Answer:")
            st.write(answer)
            
            if sources:
                with st.expander(f"Sources ({len(sources)})"):
                    for i, source in enumerate(sources, 1):
                        score = source.get("similarity_score", 0)
                        filename = source.get("source_file", f"Document_{i}")
                        st.write(f"**Source {i}: {filename}** (Relevance: {score:.3f})")
                        st.write(source.get("text", "")[:300] + "...")
            
            st.session_state.history.append({
                "question": question,
                "answer": answer,
                "sources": sources
            })

if st.session_state.history:
    st.header("History")
    
    history_text = "StudyMate Q&A History\n" + "="*40 + "\n\n"
    for entry in st.session_state.history:
        history_text += f"Q: {entry['question']}\n\nA: {entry['answer']}\n\n---\n\n"
    
    st.download_button(
        "Download History",
        data=history_text,
        file_name=f"studymate_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime='text/plain'
    )
    
    for entry in reversed(st.session_state.history):
        with st.expander(f"Q: {entry['question'][:50]}..."):
            st.write(f"**Q:** {entry['question']}")
            st.write(f"**A:** {entry['answer']}")