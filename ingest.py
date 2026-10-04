from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb


PDF_PATH = "pps.case study.pdf"
DB_PATH = "./rag_database"


# -----------------------------
# 1. Read PDF
# -----------------------------

reader = PdfReader(PDF_PATH)

text = ""

for page in reader.pages:

    page_text = page.extract_text()

    if page_text:
        text += page_text + "\n"


# -----------------------------
# 2. Chunk text
# -----------------------------

chunk_size = 700
chunk_overlap = 100

chunks = []

start = 0

while start < len(text):

    end = start + chunk_size

    chunk = text[start:end].strip()

    if chunk:
        chunks.append(chunk)

    start += chunk_size - chunk_overlap


print("Chunks:", len(chunks))


# -----------------------------
# 3. Embedding model
# -----------------------------

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# -----------------------------
# 4. Create embeddings
# -----------------------------

embeddings = model.encode(chunks)


# -----------------------------
# 5. ChromaDB
# -----------------------------

client = chromadb.PersistentClient(
    path=DB_PATH
)

collection = client.get_or_create_collection(
    name="documents"
)


# -----------------------------
# 6. Store
# -----------------------------

ids = [
    f"chunk_{i}"
    for i in range(len(chunks))
]

collection.add(
    ids=ids,
    documents=chunks,
    embeddings=embeddings.tolist()
)


print(
    "Stored:",
    collection.count(),
    "chunks"
)