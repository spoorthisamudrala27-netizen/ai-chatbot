from rag_pipeline import rag_pipeline


question = input(
    "Ask a question about your PDF: "
)


answer, documents = rag_pipeline(
    question
)


print("\n")
print("ANSWER")
print("=" * 50)

print(answer)


print("\n")
print("RETRIEVED DOCUMENTS")
print("=" * 50)

for i, document in enumerate(
    documents,
    start=1
):

    print(
        f"\n--- Chunk {i} ---"
    )

    print(document)