import os
import shutil
import chromadb

from pypdf import PdfReader

from llama_index.core import (
    Document,
    VectorStoreIndex,
    StorageContext,
)

from llama_index.core.node_parser import (
    SentenceSplitter,
)

from llama_index.vector_stores.chroma import (
    ChromaVectorStore,
)

from llama_index.embeddings.huggingface import (
    HuggingFaceEmbedding,
)

# -----------------------------
# Config
# -----------------------------
UPLOAD_FOLDER = "uploads"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "company_docs"

# -----------------------------
# Create folders
# -----------------------------
os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

# -----------------------------
# Load PDFs
# -----------------------------
documents = []

total_files = 0
total_characters = 0

print("📂 Scanning uploads folder...")

pdf_files = [
    f for f in os.listdir(UPLOAD_FOLDER)
    if f.lower().endswith(".pdf")
]

if not pdf_files:
    raise ValueError(
        "No PDF files found."
    )

for file_name in pdf_files:

    file_path = os.path.join(
        UPLOAD_FOLDER,
        file_name
    )

    print(f"\n📄 Processing: {file_name}")

    try:

        reader = PdfReader(file_path)

        text = ""

        for page in reader.pages:
            text += page.extract_text() or ""

        if not text.strip():
            continue

        documents.append(
            Document(
                text=text,
                metadata={
                    "file_name": file_name
                }
            )
        )

        total_files += 1
        total_characters += len(text)

        print(
            f"✓ Extracted {len(text)} characters"
        )

    except Exception as e:

        print(
            f"❌ Error reading {file_name}"
        )

        print(e)

print("\n----------------------------")
print(f"PDFs Indexed: {total_files}")
print(f"Characters: {total_characters}")
print("----------------------------")

# -----------------------------
# Embedding Model
# -----------------------------
print("\n🤖 Loading embedding model...")

embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# -----------------------------
# Chunking
# -----------------------------
print("\n✂️ Creating chunks...")

# UPDATED: Increased chunk parameters to map multi-document cross references perfectly
splitter = SentenceSplitter(
    chunk_size=768,
    chunk_overlap=200
)

nodes = splitter.get_nodes_from_documents(
    documents
)

print(
    f"📦 Total Chunks Created: {len(nodes)}"
)

# -----------------------------
# Rebuild Chroma
# -----------------------------
print("\n🗑️ Rebuilding Vector Database...")

if os.path.exists(CHROMA_PATH):
    shutil.rmtree(CHROMA_PATH)

client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = client.get_or_create_collection(
    COLLECTION_NAME
)

vector_store = ChromaVectorStore(
    chroma_collection=collection
)

storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)

# -----------------------------
# Build Index
# -----------------------------
print("\n⚡ Creating Vector Index...")

VectorStoreIndex(
    nodes=nodes,
    storage_context=storage_context,
    embed_model=embed_model,
)

print("\n✅ Index created successfully!")

print(
    f"📚 Total Documents: {total_files}"
)

print(
    f"📦 Total Chunks: {len(nodes)}"
)

print(
    f"📝 Total Characters: {total_characters}"
)