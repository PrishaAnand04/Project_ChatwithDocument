from flask import Flask, render_template, request, jsonify
from openai import AzureOpenAI
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_openai import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from bs4 import BeautifulSoup
import requests
import dotenv 
from langchain_openai import AzureChatOpenAI
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough
import fitz
from werkzeug.utils import secure_filename
import os
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


dotenv.load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

global_text=""

@app.route('/')
def index():
    return render_template('front_end.html')

def extract_from_url(url):
    url_add=url
    respon=requests.get(url_add)
    soup=BeautifulSoup(respon.content, 'html.parser')
    paras=soup.find_all('p')
    text='\n'.join([p.get_text() for p in paras])
    return text

def extract_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, max_tokens):
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
    prompt = f"Summarize the following text:\n\n{text}"

    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
    return summary
    

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
        if('file' in request.files):
            global_text= file_up()
            chunks=chunk_text(global_text,max_tokens=1500)
            summary_f=[summary_g(chunk) for chunk in chunks]
            summary=' '.join(summary_f)
            return jsonify({"summary": summary})
        elif('url' in request.json !=""):
            global_text=url()
            chunks=chunk_text(global_text, max_tokens=1500)
            summary_url=[summary_g(chunk) for chunk in chunks]
            summary=' '.join(summary_url)
            return jsonify({"summary":summary})
        else:
            return jsonify({"error": "No file or URL provided"}), 400
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
def process_db(text):
    embedder = SentenceTransformer('all-MiniLM-L6-v2')  # Efficient embedding model
    chunks=chunk_text(text,500)

    chunk_embeddings = embedder.encode(chunks, convert_to_tensor=True).cpu().detach().numpy()

    dimension = chunk_embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(chunk_embeddings)
    return chunks,faiss_index,embedder

def get_most_relevant_chunk(question, chunks, index, embedder, top_k=3):
    # Embed the question
    question_embedding = embedder.encode([question])
    
    # Search for the top_k most similar chunks
    distances, closest_indices = index.search(question_embedding, top_k)
    return [chunks[i] for i in closest_indices[0]]


def answer_g(question):
    chunks,faiss_index,embedder=process_db(global_text)
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
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
    try:
        data = request.get_json()
        question= data["question"]
        answer=answer_g(question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": "Input Text is too long, can't generate the answer due to size limit for the max_tokens"}), 500
    
if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True,use_reloader=False)