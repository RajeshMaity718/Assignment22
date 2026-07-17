from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load the text document
loader = TextLoader("documents/sample.txt", encoding="utf-8")
documents = loader.load()

# Split the document into chunks
text_splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

docs = text_splitter.split_documents(documents)

# Load Hugging Face Embedding Model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create FAISS Vector Store
vectorstore = FAISS.from_documents(docs, embeddings)

print("Vector Store Created Successfully!")

# Ask user for a query
query = input("\nEnter your question: ")

# Perform Similarity Search
results = vectorstore.similarity_search(query, k=3)

print("\nTop 3 Similar Documents:\n")

for i, doc in enumerate(results, start=1):
    print(f"Result {i}")
    print(doc.page_content)
    print("-" * 50)
