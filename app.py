import streamlit as st
import os
import shutil
import subprocess
import glob

# Create uploads directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Company Knowledge Base Q&A",
    page_icon="📚",
    layout="wide"
)

# -----------------------------
# Volatile Session State Initialization
# -----------------------------
# Every time the page is refreshed, session_state is cleared.
# This forces the user to start completely fresh and re-upload documents.
if "messages" not in st.session_state:
    st.session_state.messages = []
if "index_built" not in st.session_state:
    st.session_state.index_built = False

# -----------------------------
# Reset Functions
# -----------------------------
def reset_knowledge_base():
    if os.path.exists("chroma_db"):
        shutil.rmtree("chroma_db")
    if os.path.exists("uploads"):
        shutil.rmtree("uploads")
    os.makedirs("uploads", exist_ok=True)
    st.session_state.index_built = False

# -----------------------------
# Sidebar Configuration (Fixed Indentation & Unique Keys)
# -----------------------------
with st.sidebar:
    st.header("📂 Document Management")

    uploaded_files = st.file_uploader(
        "Upload PDF Documents",
        type=["pdf"],
        accept_multiple_files=True,
        key="pdf_uploader"
    )

    if uploaded_files:
        existing_files = [os.path.basename(f) for f in glob.glob("uploads/*.pdf")]
        new_file_names = [f.name for f in uploaded_files]
        
        # If the user uploads a completely different batch, clear out the stale directory
        if set(existing_files) != set(new_file_names):
            if os.path.exists("uploads"):
                shutil.rmtree("uploads")
            os.makedirs("uploads", exist_ok=True)
            
            # Save the current fresh batch immediately
            for uploaded_file in uploaded_files:
                file_path = os.path.join("uploads", uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
            st.session_state.index_built = False

        # Correctly render button based strictly on the indexing session state flag
        if not st.session_state.index_built:
            st.warning("⚠️ New batch of documents detected. Please build the index to activate them.")
            if st.button("⚡ Build/Sync Vector Index", type="primary", key="btn_build_index"):
                with st.spinner("Processing documents into multi-chunks..."):
                    result = subprocess.run(
                        ["python", "ingest.py"],
                        capture_output=True,
                        text=True
                    )

                if result.returncode == 0:
                    st.success("Index built successfully!")
                    st.session_state.index_built = True
                    st.rerun()
                else:
                    st.error("Failed to compile database index.")
                    st.code(result.stderr)
        else:
            st.success("✨ Active vector index is loaded and ready.")

    st.divider()
    st.header("⚙️ Settings")

    if st.button("🗑️ New Chat", key="btn_new_chat"):
        st.session_state.messages = []
        st.rerun()

    if st.button("⚠️ Reset Knowledge Base", key="btn_reset_kb"):
        reset_knowledge_base()
        st.session_state.messages = []
        st.success("Knowledge Base Reset Successfully!")
        st.rerun()

# -----------------------------
# Main Application Content
# -----------------------------
st.title("📚 Company Knowledge Base Q&A")
st.caption("Ask questions from your uploaded company documents.")

# Lazy load search modules to prevent initialization crashes
from query_engine import retrieve_context
from chat_engine import generate_answer

# Display Chat History safely
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# User Question Input Vector
question = st.chat_input("Ask a question...")

if question:
    # Immediately render and preserve user query in history logs
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    try:
        with st.spinner("Searching structural documents..."):
            context, nodes = retrieve_context(question)

        if not nodes or len(nodes) == 0:
            answer = "I could not find relevant context information inside the provided organizational documents."
        else:
            with st.spinner("Synthesizing context-aware response..."):
                answer = generate_answer(question, context)

        # Append and render assistant responses safely
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.write(answer)

        # Render context citations cleanly if matches are found
        if nodes and len(nodes) > 0:
            st.subheader("📄 Reference Sources")
            for i, node in enumerate(nodes, start=1):
                source_name = node.metadata.get("file_name", "Unknown File")
                with st.expander(f"Source {i} — {source_name}"):
                    if hasattr(node, 'score') and node.score is not None:
                        st.write(f"**Distance Metric Score:** `{node.score:.4f}`")
                    st.markdown(f"*{node.text.strip()[:1000]}*")

    except Exception as e:
        st.error("An operational pipeline exception occurred while processing this message.")
        st.exception(e)