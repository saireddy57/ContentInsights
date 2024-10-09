from langchain_community.embeddings import HuggingFaceEmbeddings
import whisper
import torch
# model = HuggingFaceEmbeddings("nomic-ai/nomic-embed-text-v1.5")
model_kwargs = {'device': 'cpu','trust_remote_code': True}
encode_kwargs = {'normalize_embeddings': False}

class Config:
    model = HuggingFaceEmbeddings(          
            model_name="nomic-ai/nomic-embed-text-v1.5",
            # model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    whisper_model = whisper.load_model("tiny", device=device)
    # allowed_file_types = ["doc", "docx", "pdf", "jpg", "png", "csv", "xlsx"]
    allowed_file_types = ["pdf","doc", "docx", "jpg", "png"]

