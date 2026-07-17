from dotenv import load_dotenv
import os

from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

# Load API key from .env file
load_dotenv()

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

# Create OpenAI Embedding Model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

print("\nEmbedding Results:\n")

# Convert each chunk into embedding
for i, chunk in enumerate(chunks, start=1):
    vector = embeddings.embed_query(chunk)

    print(f"Chunk {i}")
    print("Embedding Length:", len(vector))
    print("First 10 Values:")
    print(vector[:10])
    print("=" * 50)
