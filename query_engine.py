import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter

def retrieve_context(question):
    # Initialize the same embedding model used during ingestion
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )

    # Connect to the persistent ChromaDB instance
    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    # Get the existing document collection
    collection = client.get_collection(
        "company_docs"
    )

    # Wrap the collection in LlamaIndex's ChromaVectorStore
    vector_store = ChromaVectorStore(
        chroma_collection=collection
    )

    # SYNC FIX: Explicitly match the exact chunk parameters used in ingest.py
    # This guides the vector index on how to map your multi-PDF vector space.
    transformations = [
        SentenceSplitter(chunk_size=768, chunk_overlap=200)
    ]

    # Reconstruct the index from the vector store
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model,
        transformations=transformations
    )

    # Cast a wider retrieval net to capture distinct pages across multi-PDF uploads
    retriever = index.as_retriever(
        similarity_top_k=15
    )

    # Fetch relevant text chunks matching the user's question
    nodes = retriever.retrieve(question)

    # Take the top 5 highest-ranking sorted nodes directly
    filtered_nodes = nodes[:5]

    print(f"\n========== Multi-Doc Retrieval Results ({len(filtered_nodes)} Chunks Kept) ==========")
    for node in filtered_nodes:
        source_doc = node.metadata.get("file_name", "Unknown")
        score = getattr(node, "score", "N/A")
        print(f"📄 Doc: {source_doc} | Similarity Score: {score}")
        print(f"Snippet: {node.text[:150]}...")
        print("-------------------------------------------------------------------")
    print("====================================================================\n")

    # Join the text fragments into a single string for the LLM prompt context
    context = "\n\n".join(
        node.text for node in filtered_nodes
    )

    return context, filtered_nodes