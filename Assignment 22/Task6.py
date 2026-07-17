from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter

# Read document
with open("documents/sample.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Split into chunks
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=100,
    chunk_overlap=20
)

chunks = text_splitter.split_text(text)

# Load Ollama Embedding Model
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

print("Embedding Results:\n")

for i, chunk in enumerate(chunks, start=1):
    vector = embeddings.embed_query(chunk)

    print(f"Chunk {i}")
    print("Embedding Length:", len(vector))
    print("First 10 Values:")
    print(vector[:10])
    print("=" * 50)
