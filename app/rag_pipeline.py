import os
from typing import List, Dict, Tuple
from openai import OpenAI


class RAGPipeline:
    def __init__(self, api_key: str = None):
        # Detect which API key is present in environment
        gemini_key = os.getenv("GEMINI_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")
        
        if gemini_key:
            self.provider = "gemini"
            self.client = OpenAI(
                api_key=gemini_key,
                base_url="https://generativelanguage.googleapis.com/v1beta/"
            )
            self.model = "gemini-2.5-flash"
        elif openai_key:
            self.provider = "openai"
            self.client = OpenAI(api_key=openai_key)
            self.model = "gpt-4o-mini"
        elif groq_key:
            self.provider = "groq"
            self.client = OpenAI(
                api_key=groq_key,
                base_url="https://api.groq.com/openai/v1"
            )
            self.model = "llama-3.3-70b-versatile"
        else:
            raise ValueError(
                "No LLM API key found. Please set GEMINI_API_KEY, "
                "OPENAI_API_KEY, or GROQ_API_KEY in your environment."
            )
    
    def generate_answer(self, question: str, context: str) -> str:
        """
        Generate answer using the selected LLM provider with context
        
        Args:
            question: User question
            context: Relevant document context from vector store
        
        Returns:
            Generated answer
        """
        system_prompt = """You are a helpful assistant that answers questions based on provided documents. 
        Answer the question accurately based on the context provided. 
        If the information is not in the context, say "I don't have information about this in the documents."
        Always be accurate and cite relevant information."""
        
        user_message = f"""Context from documents:
{context}

Question: {question}

Please provide a comprehensive answer based on the context above."""
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    
    def format_context(self, search_results: List[Dict]) -> str:
        """
        Format search results into context string
        
        Args:
            search_results: List of search results from vector store
        
        Returns:
            Formatted context string
        """
        if not search_results:
            return "No relevant documents found."
        
        context_parts = []
        for i, result in enumerate(search_results, 1):
            page = result.get("page", "Unknown")
            text = result.get("text", "")
            context_parts.append(f"[Page {page}] {text}")
        
        return "\n\n".join(context_parts)
    
    def process_question(self, question: str, search_results: List[Dict]) -> Tuple[str, List[Dict]]:
        """
        Process a question using RAG
        
        Args:
            question: User question
            search_results: Relevant chunks from vector store
        
        Returns:
            Tuple of (answer, formatted_sources)
        """
        # Format context
        context = self.format_context(search_results)
        
        # Generate answer
        answer = self.generate_answer(question, context)
        
        # Format sources
        sources = [
            {
                "page": result.get("page", 0),
                "text": result.get("text", "")
            }
            for result in search_results
        ]
        
        return answer, sources
