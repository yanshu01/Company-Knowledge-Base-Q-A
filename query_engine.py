import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def retrieve_context(question):
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )

    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    collection = client.get_collection(
        "company_docs"
    )

    vector_store = ChromaVectorStore(
        chroma_collection=collection
    )

    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model,
    )

    # FIX: Cast a wider net (top_k=15) so chunks from both PDF 1 and PDF 2 are pulled
    retriever = index.as_retriever(
        similarity_top_k=15
    )

    nodes = retriever.retrieve(question)

    # llama_index's Chroma integration returns a similarity score where
    # HIGHER = more relevant (it's already converted from Chroma's raw
    # distance internally), and retriever.retrieve() already returns nodes
    # sorted best-first. So we just take the top 5 directly — no cutoff,
    # no re-sorting needed. (Don't use SimilarityPostprocessor with a
    # cutoff > 1.0 here, since these scores typically range roughly 0-1.)
    filtered_nodes = nodes[:5]

    print(f"\n========== Multi-Doc Retrieval Results ({len(filtered_nodes)} Chunks Kept) ==========")
    for node in filtered_nodes:
        source_doc = node.metadata.get("file_name", "Unknown")
        score = getattr(node, "score", "N/A")
        print(f"📄 Doc: {source_doc} | Similarity: {score}")
        print(f"Snippet: {node.text[:150]}...")
        print("-------------------------------------------------------------------")
    print("====================================================================\n")

    context = "\n\n".join(
        node.text for node in filtered_nodes
    )

    return context, filtered_nodes