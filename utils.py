# utils.py

import os
import fitz # PyMuPDF
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from ibm_watsonx_ai.foundation_models import ModelInference

# Load environment variables from .env file
load_dotenv()

# --- 1. PDF Processing and Chunking ---
def process_pdfs(pdf_files):
    """
    Extracts text from uploaded PDF files and splits it into overlapping chunks.
    """
    all_text = ""
    for pdf_file in pdf_files:
        with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
            all_text += "".join(page.get_text() for page in doc) # [cite: 149]
    
    # Split text into chunks using a sliding window approach [cite: 158]
    words = all_text.split()
    chunk_size = 500 # [cite: 156]
    overlap = 100 # [cite: 156]
    
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
            
    print(f"Successfully created {len(chunks)} chunks from {len(pdf_files)} PDF(s).")
    return chunks

# --- 2. Semantic Search Engine (Embedding & FAISS) ---
def create_vector_store(text_chunks):
    """
    Creates a FAISS vector store from text chunks.
    """
    if not text_chunks:
        return None, None
        
    # Load the embedding model [cite: 76, 179]
    model_name = 'all-MiniLM-L6-v2'
    encoder = SentenceTransformer(model_name)
    
    # Generate embeddings for each chunk [cite: 181]
    embeddings = encoder.encode(text_chunks, convert_to_tensor=False)
    
    # Create a FAISS index [cite: 184]
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension) # [cite: 185]
    index.add(np.array(embeddings))
    
    print("FAISS index created successfully.")
    return index, encoder

# --- 3. LLM Integration and Answer Generation ---
# In utils.py, replace the entire get_rag_response function with this:

def get_rag_response(query, faiss_index, encoder, text_chunks):
    """
    Retrieves relevant context and generates an answer using the IBM Watsonx LLM.
    """
    # Embed the user's query
    query_vector = encoder.encode([query])
    
    # Search the FAISS index for the top k similar chunks
    k = 3
    distances, indices = faiss_index.search(np.array(query_vector), k)
    
    # Retrieve the relevant chunks
    # De-duplicate the retrieved chunks while preserving order
    unique_indices = list(dict.fromkeys(indices[0]))
    retrieved_chunks = [text_chunks[i] for i in unique_indices]
    context = "\n\n".join(retrieved_chunks)
    
    # --- Prompt Construction ---
    prompt_template = f"""
    Answer the question based strictly on the following context. If the answer is not in the context, say "I cannot answer this based on the provided documents."

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    # --- IBM Watsonx LLM Invocation ---
    try:
        # UPDATED: Changed to the non-deprecated model
        model_id = "ibm/granite-3-3-8b-instruct" 
        
        credentials = {
            "url": os.getenv("IBM_URL"),
            "apikey": os.getenv("IBM_API_KEY")
        }
        
        params = {
            "decoding_method": "greedy",
            "max_new_tokens": 300,
            "temperature": 0.5,
        }
        
        model = ModelInference(
            model_id=model_id,
            params=params,
            credentials=credentials,
            project_id=os.getenv("IBM_PROJECT_ID")
        )
        
        response = model.generate(prompt=prompt_template)
        
        # UPDATED: Changed how the generated text is accessed from the response dictionary
        generated_text = response['results'][0]['generated_text']
        return generated_text, retrieved_chunks

    except Exception as e:
        print(f"Error during LLM call: {e}")
        return "An error occurred while generating the answer.", []