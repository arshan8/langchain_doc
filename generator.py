#why use genrator ?
#does not overhwelming my large document, load document part by part and operate , helpful in chunking and ebdding large documents

#by defalut we have lazy loading in langchain, so we can use generator to load the document part by part

from pypdf import PdfReader
from uuid import uuid4
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
import hashlib

# === 1. Lazy PDF Chunker ===
def chunk_pdf_lazy(path, words_per_chunk=200):
    reader = PdfReader(path)
    words = []

    for page in reader.pages:
        text = page.extract_text()
        if not text:
            continue
        words.extend(text.split())

        while len(words) >= words_per_chunk:
            yield ' '.join(words[:words_per_chunk])
            words = words[words_per_chunk:]

    if words:
        yield ' '.join(words)

# === 2. Dummy Embedder (replace with real model call) ===
def embed(text: str):
    # Simple deterministic 384-dim float vector using hash
    h = int(hashlib.sha256(text.encode()).hexdigest(), 16)
    return [(h >> i) % 1_000 / 1_000 for i in range(384)]

# === 3. Store in Qdrant ===
def store(embedding, chunk_text, client, collection="docs"):
    client.upsert(
        collection_name=collection,
        points=[
            PointStruct(
                id=str(uuid4()),
                vector=embedding,
                payload={"text": chunk_text}
            )
        ]
    )

# === 4. Create Qdrant collection (if needed) ===
def setup_collection(client, collection_name):
    from qdrant_client.http.models import VectorParams, Distance
    if not client.collection_exists(collection_name):
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )

# === 5. Run Full Pipeline ===
def process_pdf(path, collection="docs"):
    client = QdrantClient(host="localhost", port=6333)
    setup_collection(client, collection)

    for i, chunk in enumerate(chunk_pdf_lazy(path)):
        emb = embed(chunk)
        store(emb, chunk, client, collection)
        print(f"✅ Inserted chunk {i+1}")

    print("🎉 Done embedding & storing all chunks!")

# === 🔁 Run it ===
if __name__ == "__main__":
    process_pdf("your_large_file.pdf")
