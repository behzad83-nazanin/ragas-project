import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

MODEL_ID = os.getenv("MODEL_ID", "google/flan-t5-base")


EMBED_MODEL = os.getenv(
    "EMBED_MODEL",
    "BAAI/bge-m3"
)


class CustomEmbeddings:
    def __init__(self, model_name=EMBED_MODEL):
        self.model = SentenceTransformer(
            model_name,
            trust_remote_code=True
        )

    def embed_documents(self, texts):
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embeddings.tolist()

    def embed_query(self, text):
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        return embedding.tolist()