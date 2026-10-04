import streamlit as st

from rag_pipeline import rag_pipeline


# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="Chat With PDF",
    page_icon="📚"
)


# ============================================
# TITLE
# ============================================

st.title("📚 Chat With Your PDF")

st.write(
    "Ask questions based on your document."
)


# ============================================
# CHAT HISTORY
# ============================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# ============================================
# DISPLAY OLD MESSAGES
# ============================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================
# USER INPUT
# ============================================

question = st.chat_input(
    "Ask something about your PDF..."
)


# ============================================
# PROCESS QUESTION
# ============================================

if question:

    # -----------------------------
    # Display user question
    # -----------------------------

    with st.chat_message("user"):

        st.markdown(question)


    # Save question

    st.session_state.messages.append({

        "role": "user",

        "content": question

    })


    # -----------------------------
    # Generate answer
    # -----------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching the document..."
        ):

            answer, documents = (
                rag_pipeline(question)
            )


        st.markdown(answer)


    # Save answer

    st.session_state.messages.append({

        "role": "assistant",

        "content": answer

    })