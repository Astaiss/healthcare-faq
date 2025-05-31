from transformers import AutoTokenizer, AutoModel
from pymongo import MongoClient
from dotenv import load_dotenv
import os
import re
import torch
import torch.nn.functional as F


load_dotenv()

model_name = "intfloat/multilingual-e5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

client = MongoClient(os.getenv("MONGO_URI"))
db = client["faqbot_db"]
collection = db["faq_collection"]

# Function to split text into paragraphs
def split_into_paragraphs(text):
    paragraphs = re.split(r'\n\s*\n', text.strip())
    return [p.strip() for p in paragraphs if p.strip()]


def get_embedding(text, prefix="passage: "):
    text = prefix + text.strip()
    encoded = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        model_output = model(**encoded)
    embeddings = model_output.last_hidden_state
    attention_mask = encoded["attention_mask"]
    mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    summed = torch.sum(embeddings * mask_expanded, 1)
    counted = torch.clamp(mask_expanded.sum(1), min=1e-9)
    mean_pooled = summed / counted
    return F.normalize(mean_pooled, p=2, dim=1)[0].tolist()



def embed_and_store_paragraphs(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    paragraphs = split_into_paragraphs(text)

    for para in paragraphs:
        vector = get_embedding(para, prefix="passage: ")
        doc = {
            "text": para,
            "embedding": vector
        }
        collection.insert_one(doc)

    print(f"Inserted {len(paragraphs)} paragraphs into MongoDB.")


embed_and_store_paragraphs("faq.txt")

