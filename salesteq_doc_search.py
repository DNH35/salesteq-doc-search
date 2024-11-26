from langchain.vectorstores import Qdrant
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from qdrant_client import QdrantClient
from flask import Flask, request, render_template, redirect, url_for
import os
import tempfile
import shutil
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_nvidia_ai_endpoints import ChatNVIDIA
import fitz
import base64
from PIL import Image
from io import BytesIO
import requests
import json
import langchain.schema.document
import uuid
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

API_KEY = os.getenv('API_KEY')
QDRANT_API_KEY = os.getenv('QDRANT_API_KEY')
QDRANT_HOST = os.getenv('QDRANT_HOST')

COLLECTION_NAME = "QDRANT_DOC_COLLECTION"

EMBEDDING_MODEL = NVIDIAEmbeddings(api_key=API_KEY)
QDRANT_CLIENT = QdrantClient(url=QDRANT_HOST, api_key=QDRANT_API_KEY)

app = Flask(__name__)

def initialize_qdrant_collection(client, collection_name, vector_size):
    collections = [col.name for col in client.get_collections().collections]
    if collection_name in collections:
        print(f"Collection `{collection_name}` already exists.")
    else:
        client.create_collection(
            collection_name=collection_name,
            vector_size=vector_size,
            distance="Cosine",
        )
        print(f"Created collection `{collection_name}`.")

def load_llm(model="ai-llama3-70b"):
    llm = ChatNVIDIA(
        model=model,
        api_key=API_KEY,
        temperature=0.2,
        top_p=0.7,
        max_tokens=1024,
    )
    return llm

def encode_extract_images_from_doc(file):
    pdf_file = fitz.open(file)

    image_lists = []
    for page_index in range(len(pdf_file)):
        page = pdf_file.load_page(page_index)
        image_list = page.get_images(full=True)

        if image_list:
            for image in image_list:
                image_lists.append(image)
        else:
            print("[!] No images found on page", page_index)

    processed_image_list = []
    image_base64_list = []
    for image in image_lists:
        xref = image[0]
        base_image = pdf_file.extract_image(xref)
        image_bytes = base_image["image"]
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        image = Image.open(BytesIO(image_bytes))
        image_base64_list.append(image_base64)
        processed_image_list.append(image)

    return image_base64_list

def convert_image_to_metadata(image_base64_list):
    invoke_url = "https://ai.api.nvidia.com/v1/gr/meta/llama-3.2-11b-vision-instruct/chat/completions"
    stream = True

    headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Accept": "text/event-stream" if stream else "application/json"
    }

    image_summaries = []

    for image_base64 in image_base64_list:
        payload = {
            "model": 'meta/llama-3.2-11b-vision-instruct',
            "messages": [
            {
                "role": "user",
                "content": f'What is in this image? <img src="data:image/png;base64,{image_base64}" />'
            }
            ],
            "max_tokens": 400,
            "temperature": 1.00,
            "top_p": 1.00,
            "stream": stream,
        }

        response = requests.post(invoke_url, headers=headers, json=payload)

        if stream:
            content_response = ""
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode("utf-8")
                    if decoded_line.startswith("data: "):
                        decoded_line = decoded_line[6:]
                    try:
                        data = json.loads(decoded_line)
                        delta = data.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        content_response += content
                    except json.JSONDecodeError:
                        continue

            image_summaries.append(content_response)

        else:
            response_json = response.json()
            content = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
            image_summaries.append(content_response)
        
    return image_summaries

def extract_table_as_text_from_pdf(pdf_file):
    '''
    Extract the table content as text from the given table

    Parameters:
        pdf_file: The PDF file object
    
    Returns:
        str: The table content as text
    '''
    table_texts = []

    for page_index in range(len(pdf_file)):
        page = pdf_file.load_page(page_index)

        text = page.get_text("text")
        lines = text.split("\n")
        potential_table = "\n".join([line for line in lines if "\t" in line])

        if potential_table.strip():  # Check if the table content is non-empty
            table_texts.append(potential_table)

    return table_texts


def encode_extract_tables_from_doc(file):
    '''
    Encode and extract tables from the given the PATH to PDF file

    Parameters:
        file (str): The path to the PDF file

    Returns:
        str: The table content as text
    
    '''
    pdf_file = fitz.open(file)

    table_texts = extract_table_as_text_from_pdf(pdf_file)
    return table_texts


def generate_vector_store(documents, image_base64_list, table_texts):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_chunks = text_splitter.split_documents(documents)

    vectorstore = Qdrant(
        client=QDRANT_CLIENT,
        collection_name=COLLECTION_NAME,
        embeddings=EMBEDDING_MODEL,
    )

    vectorstore.add_documents(text_chunks)
    image_metadata = convert_image_to_metadata(image_base64_list)
    print("IMAGE METADATA: ", image_metadata)

    for e, s in zip(image_base64_list, image_metadata):
        i = str(uuid.uuid4())
        doc = langchain.schema.document.Document(
            page_content = s,
            metadata = {
                'id': i,
                'type': 'image',
                'original_content': e
            }
        )
        vectorstore.add_documents([doc])

    for table_text in table_texts:
        doc_id = str(uuid.uuid4())
        vectorstore.add_documents([
            {"id": doc_id, "text": table_text, "metadata": {"type": "table"}}
        ])

    return vectorstore

def query_retrieval_chain(query, vectorstore):
    retriever = vectorstore.as_retriever()
    llm = load_llm()

    system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    results = rag_chain.invoke({"input": query})
    sources = set([doc.metadata.get('source') for doc in results['context']])
    return results['answer'], sources

def main(input_doc_dir):
    '''
    Main function to generate a vector store from the given input document directory

    Parameters:
        input_doc_dir (str): The path to the input document directory
    '''
    initialize_qdrant_collection(QDRANT_CLIENT, "QDRANT_DOC_COLLECTION", 1024)

    for file in os.listdir(input_doc_dir):
        if file.endswith('.pdf'):
            file_path = os.path.join(input_doc_dir, file)
            loader = PyPDFLoader(file_path=file_path)
            document = loader.load()
    
            generate_vector_store(document)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload_folder', methods=['POST'])
def upload_folder():
    if 'documents' not in request.files:
        return "No files uploaded", 400

    temp_dir = tempfile.mkdtemp()

    try:
        uploaded_files = request.files.getlist('documents')

        for file in uploaded_files:
            if file and file.filename.endswith('.pdf'):
                temp_file_path = os.path.join(temp_dir, file.filename)
                file.save(temp_file_path)

                try:
                    loader = PyPDFLoader(file_path=temp_file_path)
                    documents = loader.load()
                    image_base64_list = encode_extract_images_from_doc(temp_file_path)
                    image_base64_list, table_texts = encode_extract_tables_from_doc(temp_file_path)
                    generate_vector_store(documents, image_base64_list, table_texts)

                except ValueError as e:
                    print(f"Error: {e}")

                os.remove(temp_file_path)
            else:
                return "Invalid file format. Only PDF files are allowed.", 400

        return redirect(url_for('home'))

    finally:
        shutil.rmtree(temp_dir)

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query')
    vectorstore = Qdrant(
        client=QDRANT_CLIENT,
        collection_name=COLLECTION_NAME,
        embeddings=EMBEDDING_MODEL,
    )
    llm_response, results = query_retrieval_chain(query, vectorstore)
    return render_template('search_results.html', query=query, llm_response=llm_response, results=results)

if __name__ == "__main__":
    main()
    app.run(debug=True)
