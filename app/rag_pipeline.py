import os
from typing import List, Dict, Tuple
from groq import Groq


class RAGPipeline:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.client = Groq(api_key=self.api_key)
    
    def generate_answer(self, question: str, context: str) -> str:
        """
        Generate answer using Groq with context
        
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
            model="llama-3.3-70b-versatile",
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
