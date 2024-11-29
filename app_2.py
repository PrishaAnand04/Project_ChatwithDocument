from flask import Flask, render_template, request, jsonify
from bs4 import BeautifulSoup
import requests
import dotenv
import os
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from werkzeug.utils import secure_filename
import fitz  

dotenv.load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

global_text = ""

@app.route('/')
def index():
    return render_template('front_end.html')

def extract_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    paragraphs = soup.find_all('p')
    text = '\n'.join([p.get_text() for p in paragraphs])
    return text

def extract_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, max_tokens=1000):
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        if len(current_chunk) + len(word) + 1 <= max_tokens:
            current_chunk.append(word)
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def summary_g(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']


from transformers import pipeline, AutoTokenizer

# Initialize the tokenizer and summarizer once to avoid reloading in every function call
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def chunk_text_with_tokenizer(text, max_tokens=1024):
    # Tokenize the entire text first
    tokens = tokenizer(text, return_tensors="pt", truncation=False)["input_ids"][0]
    chunks = []
    
    # Split text based on token count
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        
        # Decode tokens back to text
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        
        # Ensure the chunk is within max token length (double-check)
        if len(tokenizer(chunk_text)["input_ids"]) <= max_tokens:
            chunks.append(chunk_text)
        else:
            # If the chunk is still too large, truncate it explicitly
            truncated_chunk_tokens = chunk_tokens[:max_tokens]
            truncated_chunk_text = tokenizer.decode(truncated_chunk_tokens, skip_special_tokens=True)
            chunks.append(truncated_chunk_text)
        
        start = end

    return chunks

def summary_gg(text):
    chunks = chunk_text_with_tokenizer(text, max_tokens=1024)
    summaries = []

    for chunk in chunks:
        try:
            # Generate summary and set truncation to True in case the chunk is slightly over
            summary = summarizer(chunk, max_length=150, min_length=50, do_sample=False, truncation=True)[0]['summary_text']
            summaries.append(summary)
        except Exception as e:
            print(f"Error in summarizing chunk: {e}")
            summaries.append("Summary could not be generated for this chunk.")

    # Combine all summaries into a final summary
    return ' '.join(summaries)



def file_up():
    file = request.files['file']
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        text = extract_from_pdf(file_path)
        return text

def url():
    text_input = request.json['url']
    text = extract_from_url(text_input)
    return text

@app.route('/generate_summary', methods=['POST'])
def generate_summary():
    global global_text
    try:
        if 'file' in request.files:
            global_text = file_up()
        elif 'url' in request.json and request.json['url']:
            global_text = url()
        else:
            return jsonify({"error": "No file or URL provided"}), 400

        chunks = chunk_text(global_text, max_tokens=1000)
        summaries = [summary_gg(chunk) for chunk in chunks]
        summary = ' '.join(summaries)
        return jsonify({"summary": summary})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def process_db(text):
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    chunks = chunk_text(text, max_tokens=500)
    chunk_embeddings = embedder.encode(chunks, convert_to_tensor=True).cpu().detach().numpy()

    dimension = chunk_embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(chunk_embeddings)
    
    return chunks, faiss_index, embedder

def get_most_relevant_chunk(question, chunks, index, embedder, top_k=3):
    question_embedding = embedder.encode([question])
    distances, closest_indices = index.search(question_embedding, top_k)
    return [chunks[i] for i in closest_indices[0]]

def answer_g(question):
    chunks, faiss_index, embedder = process_db(global_text)
    qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

    relevant_chunk = get_most_relevant_chunk(question, chunks, faiss_index, embedder)[0]
    result = qa_pipeline(question=question, context=relevant_chunk)
    return result['answer']

@app.route('/find_answer', methods=['POST'])
def generate_answer():
    global global_text
    try:
        if not global_text:
            return jsonify({"error": "No text available from file or URL"}), 400

        data = request.get_json()
        question = data["question"]
        answer = answer_g(question)
        return jsonify({"answer": answer})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True, use_reloader=False)
