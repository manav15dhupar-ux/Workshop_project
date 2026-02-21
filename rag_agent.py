from gemini_wrapper import GeminiWrapper
from knowledge_base import KnowledgeBase
from typing import List, Dict

class RAGAgent:

    def __init__(
        self,
        gemini_api_key: str,
        knowledge_base: KnowledgeBase = None,
        temperature: float = 0.3
    ):

        print("🚀 Initializing RAG Agent...\n")

        # ✅ Gemini initialization
        self.llm = GeminiWrapper(
            api_key=gemini_api_key,
            model_name="gemini-2.5-flash",
            temperature=temperature
        )

        # ✅ Persona
        self.llm.set_persona(
            "You are a helpful AI assistant with access to a knowledge base. "
            "When answering questions, you ALWAYS cite the source documents you used. "
            "If you don't find relevant information in the knowledge base, you say so honestly. "
            "You are accurate, helpful, and always provide context from the documents. "
            "You never make up information - you only use what's in the provided context."
        )

        self.knowledge_base = knowledge_base

        print("✅ RAG Agent ready!")
        print("   Mode: Retrieval-Augmented Generation")
        print("   Source attribution: Enabled")
        print("   Hallucination protection: Active")
        print()

    def set_knowledge_base(self, knowledge_base: KnowledgeBase):

        self.knowledge_base = knowledge_base
        stats = knowledge_base.get_stats()

        print(f"✅ Knowledge base connected!")
        print(f"   Collection: {stats['collection_name']}")
        print(f"   Chunks available: {stats['total_chunks']}\n")

    def retrieve_context(self, query: str, top_k: int = 3) -> List[Dict]:

        if not self.knowledge_base:
            print("⚠️  No knowledge base connected!")
            return []

        results = self.knowledge_base.query(query, top_k=top_k)
        return results

    def build_prompt_with_context(self, query: str, context_chunks: List[Dict]) -> str:

        if not context_chunks:
            return f"""The user asked: "{query}"

You don't have any relevant information in your knowledge base to answer this question.
Please respond honestly that you don't have this information available, and suggest 
that the user might need to provide relevant documents or ask a different question."""

        context_text = "=== KNOWLEDGE BASE CONTEXT ===\n\n"
        context_text += "Here are relevant excerpts from the knowledge base:\n\n"

        for i, chunk in enumerate(context_chunks, 1):
            source = chunk['metadata'].get('source', 'Unknown Source')
            context_text += f"[Source {i}: {source}]\n"
            context_text += f"{chunk['text']}\n\n"

        prompt = f"""{context_text}
=== USER QUESTION ===

{query}

=== INSTRUCTIONS ===

Please answer the user's question using ONLY the information provided in the context above.

Important guidelines:
1. Cite which source(s) you used
2. If the context contains the answer, provide it clearly
3. If incomplete, explain what information is available
4. DO NOT make up information
5. Be helpful and conversational

Your answer:"""

        return prompt

    def answer(self, query: str, top_k: int = 3, verbose: bool = True) -> Dict:

        if verbose:
            print(f"\n{'='*70}")
            print(f"🔍 RAG PIPELINE STARTING")
            print(f"{'='*70}\n")
            print(f"Query: '{query}'\n")

        if verbose:
            print("Step 1/3: 🔍 Retrieving relevant context...")

        context_chunks = self.retrieve_context(query, top_k=top_k)

        if context_chunks:
            if verbose:
                print(f"   ✅ Found {len(context_chunks)} relevant chunks")
        else:
            if verbose:
                print("   ⚠️  No relevant context found")

        if verbose:
            print("\nStep 2/3: 📝 Building prompt with context...")

        prompt = self.build_prompt_with_context(query, context_chunks)

        if verbose:
            print("\nStep 3/3: 🤖 Generating answer with Gemini...")

        answer = self.llm.generate(prompt)

        result = {
            'query': query,
            'answer': answer,
            'sources': [
                {
                    'text': chunk['text'][:300] + '...' if len(chunk['text']) > 300 else chunk['text'],
                    'metadata': chunk['metadata'],
                    'similarity': chunk.get('similarity', 0)
                }
                for chunk in context_chunks
            ],
            'num_sources': len(context_chunks),
            'has_sources': len(context_chunks) > 0
        }

        return result

    def interactive_mode(self):

        print("\n🤖 INTERACTIVE RAG AGENT")
        print("Type 'quit' to exit\n")

        while True:
            try:
                question = input("You: ").strip()

                if question.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break

                if not question:
                    continue

                result = self.answer(question, verbose=False)

                print(f"\n🤖 Agent: {result['answer']}\n")

            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!")
                break