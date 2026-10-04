import chromadb

from sentence_transformers import SentenceTransformer

import ollama


# =====================================================
# CONFIGURATION
# =====================================================

DB_PATH = "./rag_database"

COLLECTION_NAME = "documents"

MODEL_NAME = "phi3"

TOP_K = 3


# =====================================================
# LOAD EMBEDDING MODEL
# =====================================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# =====================================================
# CONNECT CHROMADB
# =====================================================

client = chromadb.PersistentClient(
    path=DB_PATH
)

collection = client.get_collection(
    name=COLLECTION_NAME
)


# =====================================================
# STEP 1 — QUESTION EMBEDDING
# =====================================================

def create_query_embedding(question):

    print("\n[1] Creating question embedding...")

    embedding = embedding_model.encode(
        [question]
    )

    print("[1] Embedding created.")

    return embedding.tolist()


# =====================================================
# STEP 2 — RETRIEVAL
# =====================================================

def retrieve_documents(question):

    print("\n[2] Searching ChromaDB...")

    query_embedding = create_query_embedding(
        question
    )

    results = collection.query(

        query_embeddings=query_embedding,

        n_results=TOP_K
    )

    documents = results["documents"][0]

    print(
        f"[2] Retrieved {len(documents)} chunks."
    )

    return documents


# =====================================================
# STEP 3 — CREATE CONTEXT
# =====================================================

def create_context(documents):

    print("\n[3] Creating context...")

    context = "\n\n".join(
        documents
    )

    print("[3] Context created.")

    return context


# =====================================================
# STEP 4 — CREATE RAG PROMPT
# =====================================================

def create_prompt(
    question,
    context
):

    print("\n[4] Creating RAG prompt...")

    prompt = f"""
You are a friendly document assistant.

Answer the user's question using ONLY
the information given in the context.

Do not use outside knowledge.

If the answer is not present in the context,
say:

"I couldn't find that information in the document."

Be friendly and explain the answer clearly.

CONTEXT:
--------------------

{context}

--------------------

QUESTION:

{question}

ANSWER:
"""

    print("[4] Prompt created.")

    return prompt


# =====================================================
# STEP 5 — GENERATION
# =====================================================

def generate_answer(prompt):

    print("\n[5] Sending prompt to Ollama...")

    response = ollama.chat(

        model=MODEL_NAME,

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response[
        "message"
    ][
        "content"
    ]

    print("[5] Answer generated.")

    return answer


# =====================================================
# COMPLETE RAG PIPELINE
# =====================================================

def rag_pipeline(question):

    print("\n")
    print("=" * 60)
    print("RAG PIPELINE STARTED")
    print("=" * 60)

    # 1. Retrieval
    documents = retrieve_documents(
        question
    )

    # 2. Context
    context = create_context(
        documents
    )

    # 3. Prompt
    prompt = create_prompt(
        question,
        context
    )

    # 4. Generation
    answer = generate_answer(
        prompt
    )

    print("\n")
    print("=" * 60)
    print("RAG PIPELINE COMPLETED")
    print("=" * 60)

    return answer, documents