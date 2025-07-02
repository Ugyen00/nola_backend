import openai
from typing import List, Dict, Any
import uuid
import logging
from config import config
from vector_store import VectorStore

logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
        self.vector_store = VectorStore()
        self.conversations = {}  # In-memory storage for conversations
    
    def _create_conversation_id(self) -> str:
        """Create a unique conversation ID"""
        return str(uuid.uuid4())
    
    def _get_conversation_history(self, conversation_id: str) -> List[Dict[str, str]]:
        """Get conversation history"""
        return self.conversations.get(conversation_id, [])
    
    def _save_conversation_turn(self, conversation_id: str, user_message: str, assistant_response: str):
        """Save a conversation turn"""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        
        self.conversations[conversation_id].extend([
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_response}
        ])
    
    def _build_context_from_sources(self, sources: List[Dict[str, Any]]) -> str:
        """Build context string from retrieved sources"""
        context_parts = []
        for i, source in enumerate(sources):
            context_parts.append(f"Source {i+1}:\n{source['text']}\n")
        
        return "\n".join(context_parts)
    
    
    
    def _create_system_prompt(self, context: str) -> str:
        """Create system prompt with context"""
        return f"""You are a helpful AI assistant. Use the following context to answer the user's question. 
If the context doesn't contain relevant information to answer the question, say so and provide a general response based on your knowledge.

Context:
{context}
Instructions:
- Provide accurate and helpful responses
- If you reference information from the context, be specific
- If the context is not sufficient, acknowledge this limitation
- Be conversational and friendly
"""
    
    
    async def chat(self, message: str, conversation_id: str = None, max_tokens: int = None, temperature: float = None) -> Dict[str, Any]:
        """Process chat message and return response"""
        try:
            # Create conversation ID if not provided
            if not conversation_id:
                conversation_id = self._create_conversation_id()
            
            # Set default parameters
            max_tokens = max_tokens or config.MAX_TOKENS
            temperature = temperature or config.TEMPERATURE
            
            # Retrieve relevant documents from vector store
            sources = self.vector_store.similarity_search(message, top_k=5)
            
            # Build context from sources
            context = self._build_context_from_sources(sources)
            
            # Get conversation history
            conversation_history = self._get_conversation_history(conversation_id)
            
            # Prepare messages for OpenAI
            messages = [
                {"role": "system", "content": self._create_system_prompt(context)}
            ]
            
            # Add conversation history
            messages.extend(conversation_history)
            
            # Add current user message
            messages.append({"role": "user", "content": message})
            
            # Generate response using OpenAI
            response = self.openai_client.chat.completions.create(
                model=config.OPENAI_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            assistant_response = response.choices[0].message.content
            
            # Save conversation turn
            self._save_conversation_turn(conversation_id, message, assistant_response)
            
            # Prepare sources information
            sources_info = []
            for source in sources:
                source_info = {
                    "id": source["id"],
                    "score": round(source["score"], 4),
                    "source": source["metadata"].get("source", "unknown"),
                    "document_type": source["metadata"].get("document_type", "unknown")
                }
                sources_info.append(source_info)
            
            return {
                "response": assistant_response,
                "conversation_id": conversation_id,
                "sources": sources_info
            }
            
        except Exception as e:
            logger.error(f"Error in chat service: {str(e)}")
            raise
    
    def get_conversation_history(self, conversation_id: str) -> List[Dict[str, str]]:
        """Get full conversation history"""
        return self._get_conversation_history(conversation_id)
    
    def clear_conversation(self, conversation_id: str) -> bool:
        """Clear conversation history"""
        try:
            if conversation_id in self.conversations:
                del self.conversations[conversation_id]
                return True
            return False
        except Exception as e:
            logger.error(f"Error clearing conversation: {str(e)}")
            return False
    
    def get_active_conversations(self) -> List[str]:
        """Get list of active conversation IDs"""
        return list(self.conversations.keys())