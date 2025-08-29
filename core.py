import asyncio
import time
import io
import fitz
import numpy as np
from typing import List, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
from markitdown import MarkItDown
from sentence_transformers import SentenceTransformer

class SimplePDFProcessor:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
    
    async def extract_text_simple(self, pdf_file) -> str:
        try:
            markitdown = MarkItDown()
            
            loop = asyncio.get_event_loop()
            if hasattr(pdf_file, 'read'):
                pdf_bytes = await loop.run_in_executor(self.executor, pdf_file.read)
                stream = io.BytesIO(pdf_bytes)
                result = await loop.run_in_executor(
                    self.executor, 
                    markitdown.convert_stream, 
                    stream
                )
                return result.text_content
            
        except Exception as e:
            raise e
    
    async def create_chunks_async(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[Dict[str, Any]]:
        words = text.split()
        chunks = []
        
        start_idx = 0
        chunk_id = 0
        
        while start_idx < len(words):
            end_idx = min(start_idx + chunk_size, len(words))
            chunk_text = " ".join(words[start_idx:end_idx])
            
            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "start_word": start_idx,
                "end_word": end_idx,
                "word_count": end_idx - start_idx,
                "similarity_score": 0.0
            })
            
            start_idx = max(start_idx + chunk_size - overlap, start_idx + 1)
            chunk_id += 1
            
            if start_idx >= len(words):
                break
        
        return chunks


class SimpleVectorStore:
    def __init__(self):
        self.chunks = []
        self.embeddings = None
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.is_built = False
    
    async def add_chunks(self, chunks: List[Dict[str, Any]]):
        try:
            texts = [chunk['text'] for chunk in chunks]
            
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                self.encoder.encode,
                texts
            )
            
            self.embeddings = embeddings
            self.chunks = chunks
            self.is_built = True
            
        except Exception:
            self.chunks = chunks
            self.is_built = True
    
    async def search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        if not self.is_built or not self.chunks:
            return []
        
        try:
            loop = asyncio.get_event_loop()
            query_embedding = await loop.run_in_executor(
                None,
                self.encoder.encode,
                [query]
            )
            
            similarities = np.dot(self.embeddings, query_embedding.T).flatten()
            top_indices = np.argsort(similarities)[-k:][::-1]
            
            results = []
            for idx in top_indices:
                chunk = self.chunks[idx].copy()
                chunk['similarity_score'] = float(similarities[idx])
                results.append(chunk)
            
            return results
            
        except Exception as e:
            return self.chunks[:k]


class SimpleRAGEngine:
    def __init__(self):
        self.pdf_processor = SimplePDFProcessor()
        self.vector_store = SimpleVectorStore()
        self.is_initialized = False
    
    async def initialize_from_pdfs(self, pdf_files: List[Any]) -> bool:
        try:
            start_time = time.time()
            
            extract_tasks = [
                self.pdf_processor.extract_text_simple(pdf_file)
                for pdf_file in pdf_files
            ]
            
            texts = await asyncio.gather(*extract_tasks)
            
            chunk_tasks = [
                self.pdf_processor.create_chunks_async(text)
                for text in texts if text.strip()
            ]
            
            all_chunks_nested = await asyncio.gather(*chunk_tasks)
            
            all_chunks = []
            for file_idx, file_chunks in enumerate(all_chunks_nested):
                filename = getattr(pdf_files[file_idx], 'name', f"Document_{file_idx + 1}")
                
                for chunk in file_chunks:
                    chunk['source_file'] = filename
                    chunk['source_file_index'] = file_idx
                    chunk['global_id'] = len(all_chunks)
                    all_chunks.append(chunk)
            
            await self.vector_store.add_chunks(all_chunks)
            
            self.is_initialized = True
            
            return True
            
        except Exception as e:
            return False
    
    async def get_context_for_llm(self, question: str, k: int = 3) -> Tuple[str, List[Dict[str, Any]]]:
        if not self.is_initialized:
            return "RAG engine not initialized", []
        
        results = await self.vector_store.search(question, k)
        
        if not results:
            return "No relevant context found", []
        
        context_parts = []
        for i, result in enumerate(results, 1):
            score = result.get('similarity_score', 0)
            context_parts.append(
                f"Source {i} (Relevance: {score:.3f}):\n{result['text']}"
            )
        
        formatted_context = "\n\n".join(context_parts)
        return formatted_context, results


async def create_simple_rag_engine(pdf_files: List[Any]) -> SimpleRAGEngine:
    engine = SimpleRAGEngine()
    success = await engine.initialize_from_pdfs(pdf_files)
    
    if not success:
        raise RuntimeError("Failed to initialize RAG engine")
    
    return engine
