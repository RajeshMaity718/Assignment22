from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Read the document
with open("documents/sample.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Split the document into chunks
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=100,
    chunk_overlap=20
)

chunks = text_splitter.create_documents([text])

# Load Hugging Face Embedding Model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Store chunks in FAISS Vector Database
vector_store = FAISS.from_documents(chunks, embeddings)

print("Document stored successfully in FAISS!")

# Ask the user for a question
query = input("\nEnter your question: ")

# Search for the top 3 similar chunks
results = vector_store.similarity_search(query, k=3)

print("\nTop 3 Similar Documents:\n")

for i, doc in enumerate(results, start=1):
    print(f"Result {i}")
    print(doc.page_content)
    print("-" * 50)
