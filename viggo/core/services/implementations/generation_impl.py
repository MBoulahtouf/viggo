"""
Concrete implementations of generation services following SOLID principles.
"""

import time
from typing import List, Dict, Any, Optional

from viggo.core.services.interfaces.generation import (
    TextGenerator, PromptTemplate, GenerationService, GenerationResult, 
    GenerationContext, GenerationModel
)
from viggo.core.config import settings
from groq import Groq


class LLMTextGenerator(TextGenerator):
    """Concrete implementation of LLM text generator using Groq."""
    
    def __init__(self, model_name: str = None, temperature: float = None, max_tokens: int = None):
        self.model_name = model_name or settings.llm_model
        self.temperature = temperature or settings.llm_temperature
        self.max_tokens = max_tokens or settings.llm_max_tokens
        self.groq_client = Groq(api_key=settings.groq_api_key)
    
    def generate(self, context: GenerationContext) -> GenerationResult:
        """Generate text using LLM."""
        if not self.is_available():
            return GenerationResult(
                generated_text="LLM service is not available.",
                model_used=GenerationModel.LLM,
                confidence_score=0.0,
                metadata={"error": "LLM not available"}
            )
        
        try:
            # Create prompt from context
            prompt = self._create_prompt(context)
            
            # Generate response
            response = self.groq_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            generated_text = response.choices[0].message.content.strip()
            
            return GenerationResult(
                generated_text=generated_text,
                model_used=GenerationModel.LLM,
                confidence_score=0.8,  # Default confidence for LLM
                metadata={
                    "model": self.model_name,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                    "prompt_length": len(prompt)
                }
            )
            
        except Exception as e:
            return GenerationResult(
                generated_text=f"Error generating response: {str(e)}",
                model_used=GenerationModel.LLM,
                confidence_score=0.0,
                metadata={"error": str(e)}
            )
    
    def _create_prompt(self, context: GenerationContext) -> str:
        """Create a prompt from the generation context."""
        # Build context from retrieved content
        context_parts = []
        for i, content in enumerate(context.retrieved_content[:3]):  # Limit to top 3
            page_info = f" (Page {content.get('page', 'N/A')})" if content.get('page') else ""
            context_parts.append(f"{i+1}.{page_info} {content.get('content', '')[:300]}...")
        
        full_context = "\n\n".join(context_parts)
        
        # Create prompt
        prompt = f"""You are Viggo, a lore expert assistant. Answer the following question using the provided context from the book.

Question: {context.query}

Context:
{full_context}

Instructions:
- Provide a concise, accurate answer based on the context
- If the answer is not in the context, say so
- Include specific page references when possible
- Maintain the narrative flow and lore consistency
- Be helpful and informative

Answer:"""
        
        return prompt
    
    def get_model_type(self) -> GenerationModel:
        """Get the type of generation model."""
        return GenerationModel.LLM
    
    def is_available(self) -> bool:
        """Check if the generator is available."""
        try:
            # Test the connection
            self.groq_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            return True
        except:
            return False


class TemplateTextGenerator(TextGenerator):
    """Concrete implementation of template-based text generator."""
    
    def __init__(self, templates: Optional[Dict[str, str]] = None):
        self.templates = templates or {
            "default": "Based on the context, {query}",
            "entity_query": "The entity {entity} appears in the following context: {context}",
            "relationship_query": "The relationship between {entity1} and {entity2}: {context}"
        }
    
    def generate(self, context: GenerationContext) -> GenerationResult:
        """Generate text using templates."""
        # Determine template type based on query
        template_type = self._determine_template_type(context.query)
        template = self.templates.get(template_type, self.templates["default"])
        
        # Fill template with context
        generated_text = self._fill_template(template, context)
        
        return GenerationResult(
            generated_text=generated_text,
            model_used=GenerationModel.TEMPLATE,
            confidence_score=0.6,  # Lower confidence for template-based
            metadata={
                "template_type": template_type,
                "template_used": template
            }
        )
    
    def _determine_template_type(self, query: str) -> str:
        """Determine which template to use based on query."""
        query_lower = query.lower()
        
        if "relationship" in query_lower or "related" in query_lower:
            return "relationship_query"
        elif any(word in query_lower for word in ["who is", "what is", "where is"]):
            return "entity_query"
        else:
            return "default"
    
    def _fill_template(self, template: str, context: GenerationContext) -> str:
        """Fill template with context data."""
        # Build context string
        context_parts = []
        for content in context.retrieved_content[:2]:  # Limit to top 2
            context_parts.append(content.get('content', '')[:200])
        
        context_str = " ".join(context_parts)
        
        # Simple template filling
        filled = template.format(
            query=context.query,
            context=context_str,
            entity="the entity",  # Could be extracted from query
            entity1="entity1",    # Could be extracted from query
            entity2="entity2"     # Could be extracted from query
        )
        
        return filled
    
    def get_model_type(self) -> GenerationModel:
        """Get the type of generation model."""
        return GenerationModel.TEMPLATE
    
    def is_available(self) -> bool:
        """Check if the generator is available."""
        return True  # Template generator is always available


class RAGPromptTemplate(PromptTemplate):
    """Concrete implementation of RAG-specific prompt template."""
    
    def __init__(self):
        self.template_name = "rag_prompt"
    
    def create_prompt(self, context: GenerationContext) -> str:
        """Create a RAG-specific prompt."""
        # Separate results by source if available
        semantic_results = [r for r in context.retrieved_content if r.get('source') == 'semantic']
        keyword_results = [r for r in context.retrieved_content if r.get('source') == 'keyword']
        graph_results = [r for r in context.retrieved_content if r.get('source') == 'graph']
        
        prompt_parts = [
            "You are Viggo, a lore expert. Answer the following question using the provided context:",
            f"\nQuestion: {context.query}\n"
        ]
        
        if graph_results:
            prompt_parts.append("Structured Data (Authoritative Facts):")
            for result in graph_results[:2]:
                prompt_parts.append(f"- {result.get('content', '')}")
        
        if semantic_results:
            prompt_parts.append("\nLore Context (Narrative Understanding):")
            for result in semantic_results[:2]:
                page_info = f" (Page {result.get('page', 'N/A')})" if result.get('page') else ""
                prompt_parts.append(f"-{page_info} {result.get('content', '')[:200]}...")
        
        if keyword_results:
            prompt_parts.append("\nExact Matches (Precision):")
            for result in keyword_results[:2]:
                page_info = f" (Page {result.get('page', 'N/A')})" if result.get('page') else ""
                prompt_parts.append(f"-{page_info} {result.get('content', '')[:200]}...")
        
        prompt_parts.extend([
            "\nInstructions:",
            "- Prioritize structured data for authoritative facts",
            "- Use semantic search for narrative context",
            "- Include keyword matches for precision",
            "- Provide specific page references when possible",
            "- Synthesize a cohesive, lore-consistent answer",
            "\nAnswer:"
        ])
        
        return "\n".join(prompt_parts)
    
    def get_template_name(self) -> str:
        """Get the name of this template."""
        return self.template_name
    
    def supports_context_type(self, context_type: str) -> bool:
        """Check if this template supports the given context type."""
        return context_type in ["rag", "hybrid", "multi_source"]


class ConcreteGenerationService(GenerationService):
    """Concrete implementation of generation service."""
    
    def __init__(self):
        self.generators: Dict[GenerationModel, TextGenerator] = {}
        self.default_generator = None
    
    def add_generator(self, generator: TextGenerator) -> None:
        """Add a text generator to the service."""
        model_type = generator.get_model_type()
        self.generators[model_type] = generator
        
        # Set as default if it's the first one or if it's an LLM
        if self.default_generator is None or model_type == GenerationModel.LLM:
            self.default_generator = generator
    
    def remove_generator(self, model_type: GenerationModel) -> None:
        """Remove a generator from the service."""
        if model_type in self.generators:
            del self.generators[model_type]
            
            # Update default generator if needed
            if self.default_generator and self.default_generator.get_model_type() == model_type:
                self.default_generator = next(iter(self.generators.values()), None)
    
    def generate_response(self, context: GenerationContext) -> GenerationResult:
        """Generate a response using the best available generator."""
        if not self.generators:
            return GenerationResult(
                generated_text="No generation services available.",
                model_used=GenerationModel.TEMPLATE,
                confidence_score=0.0,
                metadata={"error": "No generators available"}
            )
        
        # Try to use the default generator first
        if self.default_generator and self.default_generator.is_available():
            return self.default_generator.generate(context)
        
        # Fall back to any available generator
        for generator in self.generators.values():
            if generator.is_available():
                return generator.generate(context)
        
        # If no generators are available, return error
        return GenerationResult(
            generated_text="All generation services are currently unavailable.",
            model_used=GenerationModel.TEMPLATE,
            confidence_score=0.0,
            metadata={"error": "All generators unavailable"}
        )
    
    def get_available_models(self) -> List[GenerationModel]:
        """Get list of available generation models."""
        return [model_type for model_type, generator in self.generators.items() if generator.is_available()]
