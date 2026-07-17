from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter

# Read the text file
with open("documents/sample.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Split the document into chunks
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=100,
    chunk_overlap=20
)

chunks = text_splitter.split_text(text)

print("Document Chunks:\n")

for i, chunk in enumerate(chunks, start=1):
    print(f"Chunk {i}:")
    print(chunk)
    print("-" * 40)

# Load Hugging Face Embedding Model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("\nEmbedding Results:\n")

# Convert each chunk into embedding
for i, chunk in enumerate(chunks, start=1):
    vector = embeddings.embed_query(chunk)

    print(f"Chunk {i}")
    print("Embedding Length:", len(vector))
    print("First 10 Values:")
    print(vector[:10])
    print("=" * 50)
