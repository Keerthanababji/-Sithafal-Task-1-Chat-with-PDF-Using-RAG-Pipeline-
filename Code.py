import os
import PyPDF2
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import openai

# Ensure OpenAI API key is set
openai.api_key = os.getenv('OPENAI_API_KEY')

# Initialize the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize FAISS index
dimension = 384  # Dimensionality for 'all-MiniLM-L6-v2'
index = faiss.IndexFlatL2(dimension)
chunks_map = []  # To map chunk indices to content

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            pages_text = [page.extract_text() for page in reader.pages]
        return pages_text
    except Exception as e:
        print(f"Error extracting text: {e}")
        return []

# Function to extract tables from a PDF
def extract_tables_from_pdf(pdf_path):
    try:
        tables = pd.read_html(pdf_path, flavor='lxml')
        return tables
    except Exception as e:
        print(f"Error extracting tables: {e}")
        return []

# Function to chunk text
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Function to embed text chunks
def embed_text(chunks):
    try:
        return embedding_model.encode(chunks, convert_to_numpy=True)
    except Exception as e:
        print(f"Error during embedding: {e}")
        return np.array([])

# Function to add embeddings to FAISS index
def add_to_index(chunks, embeddings):
    try:
        faiss_embeddings = embeddings.astype('float32')
        index.add(faiss_embeddings)
        chunks_map.extend(chunks)
    except Exception as e:
        print(f"Error adding to index: {e}")

# Function to search in FAISS index
def search_index(query, top_k=5):
    try:
        query_embedding = embedding_model.encode([query], convert_to_numpy=True).astype('float32')
        distances, indices = index.search(query_embedding, top_k)
        return [chunks_map[i] for i in indices[0]]
    except Exception as e:
        print(f"Error during search: {e}")
        return []

# Function to generate a response using OpenAI
def generate_response(query, retrieved_chunks):
    prompt = f"Query: {query}\nRelevant information:\n" + "\n".join(retrieved_chunks)
    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=500
        )
        return response['choices'][0]['text'].strip()
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Error generating response."

# Main processing function
def process_pdf_and_query(pdf_path, query):
    # Step 1: Extract text and tables
    pages_text = extract_text_from_pdf(pdf_path)
    
    # Chunk and embed text
    all_chunks = []
    for page_text in pages_text:
        all_chunks.extend(chunk_text(page_text))

    embeddings = embed_text(all_chunks)
    add_to_index(all_chunks, embeddings)

    # Step 2: Handle query
    retrieved_chunks = search_index(query)
    response = generate_response(query, retrieved_chunks)

    return response

# Example Usage
pdf_path = "tables-charts-and-graphs-with-examples-from.pdf"  # Replace with local path
query1 = "What is the unemployment information based on the type of degree on page 2?"
query2 = "Extract tabular data from page 6 and compare values."

response1 = process_pdf_and_query(pdf_path, query1)
response2 = process_pdf_and_query(pdf_path, query2)

print("Query 1 Response:\n", response1)
print("Query 2 Response:\n", response2)
