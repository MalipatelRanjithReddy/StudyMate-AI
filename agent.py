import asyncio
import logging
import time
import random
from typing import List, Dict, Any, Annotated, Optional
from typing_extensions import TypedDict
import os
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from langchain_core.tools import tool, BaseTool
from pydantic import BaseModel, Field
from langchain_ibm import ChatWatsonx
from langgraph.checkpoint.memory import MemorySaver
from core import SimpleRAGEngine

logger = logging.getLogger(__name__)

class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

def create_search_tool(rag_engine: SimpleRAGEngine):
    @tool
    def studymate_search(query: str, num_results: int = 5) -> str:
        """Search uploaded PDF documents to find relevant content for answering student questions.
        
        This tool can be called multiple times with different queries to gather comprehensive information.
        Use specific, targeted queries for better results. You can search for:
        - Specific concepts or topics
        - Definitions and explanations
        - Examples and use cases
        - Related information to build complete answers
        
        Args:
            query: The specific search query - be precise and focused
            num_results: Number of relevant chunks to return (default: 5, max: 10)
            
        Returns:
            str: Formatted search results with file names, relevance scores and content
        """
        try:
            import asyncio
            
            
            async def search_async():
                context, sources = await rag_engine.get_context_for_llm(query, k=num_results)
                
                if not sources:
                    return f"No relevant content found for query: '{query}'. Try rephrasing or using different keywords."
                
                formatted_result = f"Found {len(sources)} relevant sources for '{query}':\n\n"
                for i, source in enumerate(sources, 1):
                    score = source.get('similarity_score', 0)
                    filename = source.get('source_file', f'Document_{i}')
                    text = source.get('text', '')[:300] + "..." if len(source.get('text', '')) > 300 else source.get('text', '')
                    formatted_result += f"Source {i} - {filename} (Relevance: {score:.3f}):\n{text}\n\n"
                
                return formatted_result
            
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    import concurrent.futures
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, search_async())
                        return future.result()
                else:
                    return loop.run_until_complete(search_async())
            except RuntimeError:
                return asyncio.run(search_async())
                
        except Exception as e:
            return f"Error searching documents: {str(e)}"
    
    return studymate_search

class StudyMateSearchInput(BaseModel):
    query: str = Field(description="The student's question to search in uploaded documents")

class StudyMateAgent:
    def __init__(self, rag_engine: SimpleRAGEngine, model_id: str = "ibm/granite-13b-instruct-v2"):
        self.rag_engine = rag_engine
        
        self.llm = ChatWatsonx(
            model_id=model_id,
            url=os.environ["IBM_URL"],
            apikey=os.environ["IBM_API_KEY"],
            project_id=os.environ["IBM_PROJECT_ID"],
            params={"temperature": 0.3}
        )
        
        self.tools = [create_search_tool(rag_engine)]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()
        
        self.system_prompt = """You are StudyMate, an AI academic assistant that helps students learn from their uploaded PDF study materials.

Your capabilities:
- Search through uploaded academic PDFs using the studymate_search tool
- Perform multiple targeted searches to gather comprehensive information
- Provide clear, educational explanations based on the content
- Reference sources for verification
- Maintain academic integrity by grounding answers in uploaded content

Search Strategy:
- Use MULTIPLE tool calls when needed to gather complete information
- Start with broad searches, then narrow down with specific queries
- Search for different aspects: definitions, examples, applications, etc.
- Adjust num_results parameter based on complexity (3-7 for most queries)
- If initial search doesn't provide enough context, search again with different terms

When a student asks a question:
1. Analyze the question to identify key concepts and information needs
2. Use studymate_search strategically - make multiple calls if needed:
   - First search: broad query to understand the topic
   - Additional searches: specific concepts, examples, or related information
3. Synthesize information from all searches into a comprehensive answer
4. Provide clear, educational explanations with proper source attribution
5. If insufficient information is found, clearly state limitations and suggest alternative approaches

Remember: You can and should use the search tool multiple times per question to provide thorough, well-researched answers."""
    
    def _build_graph(self):
        graph_builder = StateGraph(State)
        
        graph_builder.add_node("agent", self.agent_node)
        graph_builder.add_node("tools", ToolNode(self.tools))
        
        graph_builder.add_edge(START, "agent")
        graph_builder.add_conditional_edges(
            "agent",
            tools_condition,
            {"tools": "tools", "__end__": END}
        )
        graph_builder.add_edge("tools", "agent")
        
        return graph_builder.compile(checkpointer=self.checkpointer)

    def _extract_sources_from_messages(self, messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        sources = []
        seen_sources = set()
        
        for message in messages:
            if hasattr(message, 'type') and message.type == 'tool':
                content = message.content
                if isinstance(content, str) and 'Source' in content:
                    lines = content.split('\n')
                    current_source = {}
                    
                    for line in lines:
                        if line.strip().startswith('Source') and '(Relevance:' in line:
                            parts = line.split(' - ')
                            if len(parts) >= 2:
                                filename_part = parts[1].split(' (Relevance: ')
                                if len(filename_part) >= 2:
                                    filename = filename_part[0]
                                    score_str = filename_part[1].replace('):', '')
                                    try:
                                        score = float(score_str)
                                    except:
                                        score = 0.0
                                    
                                    source_key = f"{filename}_{score}"
                                    if source_key not in seen_sources:
                                        current_source = {
                                            'source_file': filename,
                                            'similarity_score': score,
                                            'text': ''
                                        }
                                        seen_sources.add(source_key)
                        elif current_source and line.strip() and not line.strip().startswith('Source'):
                            if current_source['text']:
                                current_source['text'] += ' ' + line.strip()
                            else:
                                current_source['text'] = line.strip()
                    
                    if current_source and current_source not in sources:
                        sources.append(current_source)
        
        return sources

    def agent_node(self, state: State):
        logger.info("Processing student question...")
        
        messages = [SystemMessage(content=self.system_prompt)] + state["messages"]
        
        max_retries = 3
        base_delay = 1
        
        for attempt in range(max_retries + 1):
            try:
                response = self.llm_with_tools.invoke(messages)
                return {"messages": [response]}
                
            except Exception as e:
                error_str = str(e).lower()
                
                if "rate limit" in error_str or "429" in error_str:
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                        logger.warning(f"Rate limit error on attempt {attempt + 1}. Retrying in {delay:.2f} seconds...")
                        time.sleep(delay)
                        continue
                    else:
                        logger.error("Max retries reached for rate limiting")
                        raise
                
                elif any(token_error in error_str for token_error in ["token", "context", "length", "too long"]):
                    logger.warning("Token limit error detected. Clearing previous messages and retrying...")
                    
                    recent_messages = state["messages"][-2:] if len(state["messages"]) > 2 else state["messages"]
                    messages = [SystemMessage(content=self.system_prompt)] + recent_messages
                    
                    try:
                        response = self.llm_with_tools.invoke(messages)
                        return {"messages": [response]}
                    except Exception as retry_error:
                        logger.error(f"Failed even after clearing messages: {str(retry_error)}")
                        raise retry_error
                else:
                    raise

    def query(self, user_request: str, thread_id: str = "default") -> Dict[str, Any]:
        try:
            result = self.graph.invoke(
                {"messages": [HumanMessage(content=user_request)]},
                config={"configurable": {"thread_id": thread_id}, "recursion_limit": 100}
            )
            
            last_message = result["messages"][-1]
            
            tool_sources = self._extract_sources_from_messages(result["messages"])
            
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return {
                    'status': 'success',
                    'response': last_message.content,
                    'tool_calls': len(last_message.tool_calls),
                    'search_executed': True,
                    'sources': tool_sources
                }
            else:
                return {
                    'status': 'success',
                    'response': last_message.content,
                    'tool_calls': 0,
                    'search_executed': False,
                    'sources': tool_sources
                }
                
        except Exception as e:
            error_str = str(e).lower()
            
            if "rate limit" in error_str or "429" in error_str:
                logger.error(f"Rate limiting error: {str(e)}")
                return {
                    'status': 'error',
                    'error': 'Rate limit exceeded. Please try again in a few moments.',
                    'response': None,
                    'sources': []
                }
            elif any(token_error in error_str for token_error in ["token", "context", "length", "too long"]):
                logger.error(f"Token limit error: {str(e)}")
                return {
                    'status': 'error',
                    'error': 'Request too long. Please try a shorter question.',
                    'response': None,
                    'sources': []
                }
            else:
                logger.error(f"Error in query execution: {str(e)}")
                return {
                    'status': 'error',
                    'error': str(e),
                    'response': None,
                    'sources': []
                }

    def get_conversation_history(self, thread_id: str = "default") -> List[Dict]:
        try:
            state = self.graph.get_state(config={"configurable": {"thread_id": thread_id}})
            messages = state.values.get("messages", [])
            
            history = []
            for msg in messages:
                if hasattr(msg, 'type'):
                    history.append({
                        'type': msg.type,
                        'content': msg.content,
                        'timestamp': getattr(msg, 'timestamp', None)
                    })
            
            return history
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {str(e)}")
            return []

    def clear_conversation_history(self, thread_id: str = "default") -> bool:
        try:
            self.checkpointer = MemorySaver()
            self.graph = self._build_graph()
            logger.info(f"Conversation history cleared for thread: {thread_id}")
            return True
        except Exception as e:
            logger.error(f"Error clearing conversation history: {str(e)}")
            return False

    async def query_async(self, question: str) -> Dict[str, Any]:
        logger.info(f"🤖 Processing question: {question[:50]}...")
        result = self.query(question)
        logger.info(f"🔍 Query result status: {result.get('status', 'unknown')}")
        
        if result['status'] == 'success':
            logger.info(f"✅ Success! Response length: {len(result.get('response', ''))}")
            sources = result.get('sources', [])
            logger.info(f"📚 Found {len(sources)} sources from tool calls")
            return {
                "success": True,
                "answer": result['response'],
                "sources": sources
            }
        else:
            logger.error(f"❌ Error: {result.get('error', 'Unknown error')}")
            return {
                "success": False,
                "error": result.get('error', 'Unknown error'),
                "answer": f"Error: {result.get('error', 'Unknown error')}",
                "sources": []
            }

async def create_agent(rag_engine: SimpleRAGEngine, model_id: str = "ibm/granite-13b-instruct-v2") -> StudyMateAgent:
    return StudyMateAgent(rag_engine, model_id)
