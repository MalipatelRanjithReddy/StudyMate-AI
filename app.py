import streamlit as st
import datetime
from utils import process_pdfs, create_vector_store, get_rag_response

# --- Helper Function for Downloading History ---
def format_history_for_download(history):
    """Formats the session history into a downloadable string."""
    formatted_text = "StudyMate Q&A History\n"
    formatted_text += f"Exported on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    formatted_text += "="*40 + "\n\n"
    
    # Iterate in chronological order for the text file
    for entry in history:
        formatted_text += f"Q: {entry['question']}\n\n"
        formatted_text += f"A: {entry['answer']}\n\n"
        formatted_text += "---\n\n"
        
    return formatted_text

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="StudyMate",
    layout="wide"
)

st.title("📚 StudyMate: Your AI-Powered Study Assistant")
st.write("Upload your academic PDFs and ask questions to get answers directly from your study materials.")

# --- Session State Initialization ---
if "history" not in st.session_state:
    st.session_state.history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "encoder" not in st.session_state:
    st.session_state.encoder = None
if "text_chunks" not in st.session_state:
    st.session_state.text_chunks = None

# --- UI Components ---
with st.sidebar:
    st.header("Upload Your PDFs")
    uploaded_files = st.file_uploader(
        "Drag and drop one or more PDF files here",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files and st.button("Process Documents"):
        with st.spinner("Processing PDFs... This may take a moment."):
            # Step 1: Process PDFs and get chunks
            st.session_state.text_chunks = process_pdfs(uploaded_files)
            
            # Step 2: Create vector store
            st.session_state.vector_store, st.session_state.encoder = create_vector_store(st.session_state.text_chunks)
            
            st.success("Documents processed successfully! You can now ask questions.")

# --- Main Interaction Area ---
if st.session_state.vector_store is None:
    st.info("Please upload and process your PDF documents using the sidebar to begin.")
else:
    st.header("Ask a Question")
    user_query = st.text_input("Enter your question about the documents:")

    if user_query:
        with st.spinner("Searching for answers..."):
            # Step 3: Get RAG response
            answer, sources = get_rag_response(
                user_query, 
                st.session_state.vector_store, 
                st.session_state.encoder, 
                st.session_state.text_chunks
            )

            # Store the interaction in history
            st.session_state.history.append({"question": user_query, "answer": answer, "sources": sources})
            
            # Display the latest answer
            st.subheader("Answer:")
            st.write(answer)
            
            with st.expander("Referenced Paragraphs"):
                for source in sources:
                    st.info(source)
                    
# --- Q&A History Display and Download ---
if st.session_state.history:
    st.header("Q&A History")

    # Add the download button
    history_text = format_history_for_download(st.session_state.history)
    st.download_button(
        label="📥 Download Q&A History",
        data=history_text,
        file_name=f"studymate_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime='text/plain'
    )

    # Display history in reverse chronological order on the page
    for entry in reversed(st.session_state.history):
        with st.expander(f"Q: {entry['question']}"):
            st.write(f"A: {entry['answer']}")
