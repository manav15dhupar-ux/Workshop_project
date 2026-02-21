import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict
import uuid

from chunking_utility import TextChunker


class KnowledgeBase:
    """
    Vector Database Knowledge Base using ChromaDB
    """

    def __init__(self, collection_name: str = "gdg_knowledge"):

        print("🚀 Initializing Knowledge Base...")

        # In-memory ChromaDB client
        self.client = chromadb.Client()

        # Embedding function
        self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        print("   Loading embedding model: all-MiniLM-L6-v2")
        print("   (This creates 384-dimensional vectors)")

        # Create / Get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={"description": "GDG Workshop Knowledge Base"}
        )

        # Text chunker helper
        self.chunker = TextChunker(chunk_size=500, overlap=50)

        print(f"✅ Knowledge Base '{collection_name}' ready!")
        print(f"   Current documents: {self.collection.count()} chunks\n")

    # ======================================================
    # ADD SINGLE DOCUMENT
    # ======================================================

    def add_document(self, text: str, metadata: Dict = None) -> List[str]:

        if metadata is None:
            metadata = {}

        print("📄 Processing document...")

        # Chunk text
        chunks = self.chunker.chunk_text(text, method="sentences")
        print(f"   ✂️ Created {len(chunks)} chunks")

        ids = []
        texts = []
        metadatas = []

        for chunk in chunks:
            chunk_id = str(uuid.uuid4())

            ids.append(chunk_id)
            texts.append(chunk["text"])

            chunk_metadata = {
                **metadata,
                "chunk_id": chunk["chunk_id"],
                "word_count": chunk["word_count"],
                "method": chunk.get("method", "unknown"),
            }

            metadatas.append(chunk_metadata)

        print("   🧮 Generating embeddings...")

        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
        )

        print(f"✅ Added {len(chunks)} chunks")
        print(f"   Total chunks in KB: {self.collection.count()}\n")

        return ids

    # ======================================================
    # ADD MULTIPLE DOCUMENTS  ✅ NEW FIX
    # ======================================================

    def add_documents(self, texts: List[str], source: str = "Unknown") -> List[str]:
        """
        Add multiple documents to the knowledge base.
        """

        all_ids = []

        for text in texts:
            ids = self.add_document(
                text,
                metadata={"source": source}
            )
            all_ids.extend(ids)

        return all_ids

    # ======================================================
    # QUERY (Semantic Search)
    # ======================================================

    def query(self, query_text: str, top_k: int = 3) -> List[Dict]:

        print(f"🔍 Searching for: '{query_text}'")

        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k
        )

        formatted_results = []

        for i in range(len(results["ids"][0])):

            distance = results["distances"][0][i]
            similarity = 1 - distance if distance is not None else None

            formatted_results.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": distance,
                "similarity": similarity
            })

        print(f"✅ Found {len(formatted_results)} relevant chunks\n")

        return formatted_results

    # ======================================================
    # STATS
    # ======================================================

    def get_stats(self) -> Dict:

        return {
            "collection_name": self.collection.name,
            "total_chunks": self.collection.count(),
            "embedding_dimension": 384,
            "embedding_model": "all-MiniLM-L6-v2"
        }

    # ======================================================
    # CLEAR KB
    # ======================================================

    def clear(self):

        print("⚠️ Clearing knowledge base...")

        collection_name = self.collection.name

        self.client.delete_collection(collection_name)

        self.collection = self.client.create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )

        print("✅ Knowledge base cleared\n")