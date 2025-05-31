from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from pymongo import MongoClient
import re
from dotenv import load_dotenv
import os
from invoke_llm import invoke_gemini    

load_dotenv()

# Load multilingual E5 model
model_name = "intfloat/multilingual-e5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# MongoDB connection
client = MongoClient(os.getenv("MONGO_URI"))
db = client["faqbot_db"]
collection = db["faq_collection"]

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

def search_top_k(query, k=5):
    query_vector = get_embedding(query, prefix="query: ")

    results = collection.aggregate([
        {
            "$vectorSearch": {
                "index": "faq_index",              
                "path": "embedding",                
                "queryVector": query_vector,        
                "numCandidates": 100,             
                "limit": k                          
            }
        },
        {
            "$project": {
                "text": 1,
                "_id": 0
            }
        }
    ])


    return list(results)


query_text = "What kind of dental services do you offer?"
top_paragraphs = search_top_k(query_text, k=3)

# for i, para in enumerate(top_paragraphs, 1):
#     print(f"\nResult {i}:\n{para['text']}")

response = invoke_gemini(query_text, "\n".join([para['text'] for para in top_paragraphs]))
print(f"\nFAQBot Response:\n{response}")


"""
Result 1:
Dental services include cleanings, fillings, extractions, and oral cancer screenings. Pediatric dentistry ensures children’s teeth develop properly. Orthodontic consultations evaluate the need for braces or aligners. Gum disease treatment prevents tooth loss and systemic health issues. Emergency dental care is available for severe pain or trauma.

Result 2:
Our healthcare facility provides a wide range of medical services, including primary care, emergency treatment, and specialized diagnostics. We are committed to delivering high-quality, patient-centered care in a safe and compassionate environment. Appointments can be scheduled online, by phone, or in person at our reception desk. Walk-ins are accommodated based on availability, though scheduling in advance is recommended for shorter wait times. Our team consists of board-certified physicians, nurses, and specialists dedicated to meeting your healthcare needs.

Result 3:
This comprehensive overview ensures patients understand available services, policies, and advancements in healthcare. For further details, please contact our facility directly.

"""

"""
We offer cleanings, fillings, extractions, oral cancer screenings, pediatric dentistry, orthodontic consultations, gum disease treatment, and emergency dental care.
"""

"""
{
  "fields": [
    {
      "type": "vector",
      "path": "embedding",
      "numDimensions": 768,
      "similarity": "cosine"
    }
  ]
}

"""