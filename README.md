# FAQBot: Multilingual FAQ Assistant using E5 Embeddings, MongoDB Vector Search, and Gemini API

## Overview

FAQBot is an intelligent FAQ assistant that uses:

- intfloat/multilingual-e5-base for multilingual sentence embeddings

- MongoDB Atlas Vector Search for fast semantic similarity search

- Google Gemini API for generating natural language responses

This project allows users to ask questions, retrieves semantically relevant FAQ paragraphs using vector search, and generates responses based on the retrieved context.

## Features

- Multilingual support for English and other languages

- Real-time vector search using MongoDB Atlas

- Google Gemini API integration for natural language answers

## Setup Instructions

### 1. Clone the Repository
```
git clone <your-repo-url>
cd <your-repo>
```

### 2. Install Dependencies

```
pip install -r requirements.txt
```

### 3. Set up MongoDB Atlas and Vector Search

Create a MongoDB Atlas cluster

Enable Vector Search in your collection settings:
```
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
```

### 4. Create .env file in the root directory:

```
MONGO_URI=<your-mongodb-uri>
GEMINI_API_KEY=<your-gemini-api-key>
```

### 5. Prepare Data

Create a text file named faq.txt with your FAQ content (each paragraph separated by a blank line)

Example:
```
We provide dental services such as cleanings, fillings, extractions, and screenings.

Emergency dental care is available for severe pain or trauma.
```

### 6. Embed and Store FAQ Paragraphs

Run the script to embed and store paragraphs in MongoDB:
```
python store_faq.py
```

This script:

Splits text into paragraphs

- Converts each paragraph into an embedding using E5 model

- Stores the paragraph and embedding into MongoDB

## Query and Response Flow

Run the faqbot.py script to:
```
python faqbot.py
```

- Generate query embedding

- Search MongoDB for top-k similar paragraphs

- Call Gemini API with context to generate answer

```
python faqbot.py
```

## Sample Input
```
query_text = "What kind of dental services do you offer?"
```
## Sample Output
```
FAQBot Response:
We offer cleanings, fillings, extractions, oral cancer screenings, pediatric dentistry, orthodontic consultations, gum disease treatment, and emergency dental care.
```

## Project Structure

```
├── faq.txt                  
├── .env                    
├── embed_faq.py            
├── faqbot.py               
├── invoke_llm.py          
└── README.md   
```           

## Future Work

- Multilingual Query Support: Currently supports multilingual embeddings; future improvements will focus on queries in other languages (e.g., Spanish, French, German)

- Interactive Web UI using Gradio or Streamlit

- Contextual Memory using ChromaDB or Redis

- Batch Embedding Ingestion and monitoring tools

## Credits

- intfloat/multilingual-e5-base for sentence embeddings

- MongoDB Atlas Vector Search

- Google Gemini API for content generation

## License

MIT License

