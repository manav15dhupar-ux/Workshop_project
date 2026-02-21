"""
Streamlit Web Interface - Single Folder Version
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import requests
from bs4 import BeautifulSoup

# ✅ Import from same folder
from rag_agent import RAGAgent
from knowledge_base import KnowledgeBase


# ==========================================================
# SESSION STATE INIT
# ==========================================================

def init_session_state():

    if 'agent' not in st.session_state:
        st.session_state.agent = None

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'kb' not in st.session_state:
        st.session_state.kb = None


# ==========================================================
# MAIN APP
# ==========================================================

def main():

    st.set_page_config(
        page_title="GDG Knowledge Agent",
        page_icon="🤖",
        layout="wide"
    )

    init_session_state()

    st.title("🤖 GDG Knowledge Agent")
    st.markdown("*Powered by Retrieval-Augmented Generation (RAG) with Gemini AI*")
    st.markdown("---")

    # ==========================================================
    # SIDEBAR
    # ==========================================================

    with st.sidebar:

        st.header("⚙️ Configuration")

        api_key = st.text_input(
            "Gemini API Key",
            type="password",
            help="Get your key from https://aistudio.google.com/app/apikey"
        )

        # ================= INITIALIZE AGENT =================

        if st.button("🚀 Initialize Agent", use_container_width=True):

            if not api_key:
                st.error("⚠️ Please enter your Gemini API key")
            else:
                with st.spinner("Initializing RAG Agent..."):

                    try:
                        # Create Knowledge Base
                        st.session_state.kb = KnowledgeBase()

                        # Add default knowledge
                        st.session_state.kb.add_documents(
                            [
                                "GDG events are free community-driven tech events.",
                                "DevFest is the flagship annual event organized by GDG chapters.",
                                "GDG promotes learning, collaboration, and networking among developers."
                            ],
                            source="Default Knowledge"
                        )

                        # 🔍 DEBUG PRINTS
                        print("Initializing RAGAgent...")
                        print("API Key:", api_key)

                        st.session_state.agent = RAGAgent(
                            gemini_api_key=api_key,
                            knowledge_base=st.session_state.kb
                        )

                        print("Agent created successfully:", st.session_state.agent)

                        st.success("✅ Agent initialized successfully!")

                    except Exception as e:
                        st.error(f"Initialization Error: {str(e)}")
                        raise

        st.markdown("---")

        # ==========================================================
        # DOCUMENT UPLOAD
        # ==========================================================

        st.header("📄 Upload Documents")

        uploaded_files = st.file_uploader(
            "Upload .txt or .md files",
            accept_multiple_files=True,
            type=['txt', 'md']
        )

        if uploaded_files and st.button("Process Documents", use_container_width=True):

            if st.session_state.kb is None:
                st.error("⚠️ Initialize the agent first!")
            else:
                try:
                    for file in uploaded_files:
                        text = file.read().decode('utf-8')

                        st.session_state.kb.add_documents(
                            [text],
                            source=file.name
                        )

                    st.success("✅ Documents processed successfully!")

                except Exception as e:
                    st.error(f"Error processing files: {str(e)}")

        st.markdown("---")

        # ==========================================================
        # WEB SCRAPING
        # ==========================================================

        st.header("🌐 Fetch Website Data")

        url = st.text_input("Enter website URL")

        if st.button("Fetch Data", use_container_width=True):

            if st.session_state.kb is None:
                st.error("⚠️ Initialize the agent first!")
            elif not url:
                st.warning("⚠️ Please enter a URL")
            else:
                try:
                    headers = {'User-Agent': 'Mozilla/5.0'}

                    response = requests.get(url, headers=headers, timeout=10)
                    response.raise_for_status()

                    soup = BeautifulSoup(response.text, 'html.parser')

                    for script in soup(["script", "style"]):
                        script.decompose()

                    text = soup.get_text()

                    st.session_state.kb.add_documents(
                        [text],
                        source=url
                    )

                    st.success("✅ Website data added to knowledge base!")

                except Exception as e:
                    st.error(f"Error fetching website: {str(e)}")

        st.markdown("---")

        # ==========================================================
        # KNOWLEDGE STATS
        # ==========================================================

        if st.session_state.kb:
            stats = st.session_state.kb.get_stats()
            st.write("📊 Total Chunks:", stats['total_chunks'])

            if st.button("🔄 Reset Knowledge Base"):
                st.session_state.agent = None
                st.session_state.kb = None
                st.session_state.messages = []
                st.success("Knowledge base reset!")
                st.rerun()

    # ==========================================================
    # CHAT INTERFACE
    # ==========================================================

    if st.session_state.agent is None:
        st.info("👈 Initialize the agent from sidebar to start chatting.")
        return

    st.header("💬 Ask Questions")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask about GDG events, workshops, etc..."):

        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                try:
                    result = st.session_state.agent.answer(prompt, verbose=False)

                    st.markdown(result['answer'])

                    if result['sources']:
                        with st.expander(f"📚 View {len(result['sources'])} Sources"):
                            for i, source in enumerate(result['sources'], 1):
                                st.markdown(
                                    f"**Source {i}:** {source['metadata'].get('source', 'Unknown')}"
                                )
                                st.text(source['text'][:200] + "...")

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": result['answer']
                        }
                    )

                except Exception as e:
                    st.error(f"Error: {str(e)}")

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":
    main()