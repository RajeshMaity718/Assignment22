from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS, Chroma

# -----------------------------
# Step 1: Load Document
# -----------------------------
loader = TextLoader("documents/sample.txt", encoding="utf-8")
documents = loader.load()

# -----------------------------
# Step 2: Split into Chunks
# -----------------------------
splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

docs = splitter.split_documents(documents)

# -----------------------------
# Step 3: Create Embeddings
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# Step 4: Choose Vector Store
# -----------------------------

choice = input("Choose Vector Store (faiss/chroma): ").lower()

if choice == "faiss":
    vectorstore = FAISS.from_documents(docs, embeddings)

elif choice == "chroma":
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

else:
    print("Invalid Choice")
    exit()

print("\nVector Store Created Successfully!")

# -----------------------------
# Step 5: Similarity Search
# -----------------------------

query = input("\nEnter your question: ")

results = vectorstore.similarity_search(query, k=3)

print("\nTop 3 Similar Documents:\n")

for i, doc in enumerate(results, start=1):
    print(f"Result {i}")
    print(doc.page_content)
    print("-" * 50)
