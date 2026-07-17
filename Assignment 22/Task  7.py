from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load the text document
loader = TextLoader("documents/sample.txt", encoding="utf-8")
documents = loader.load()

# Split document into chunks
splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

docs = splitter.split_documents(documents)

# Load Hugging Face Embedding Model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create FAISS Vector Store
db = FAISS.from_documents(docs, embeddings)

print("FAISS Vector Store Created Successfully!")

# Save FAISS index
db.save_local("faiss_index")

print("FAISS Index Saved Successfully!")

# Reload FAISS index
new_db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

print("FAISS Index Loaded Successfully!")

# Ask user for a query
query = input("\nEnter your question: ")

# Perform similarity search
results = new_db.similarity_search(query, k=3)

print("\nTop 3 Similar Documents:\n")

for i, doc in enumerate(results, start=1):
    print(f"Result {i}")
    print(doc.page_content)
    print("-" * 50)
