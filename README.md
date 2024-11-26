# salesteq-doc-search

## Overview

This Flask-based web application uses Langchain and NVIDIA AI to provide a document and image-based retrieval system. It allows users to upload PDF documents, extract text, tables, images, and generate a vector store for querying via a question-answering system. The app utilizes Qdrant for vector storage and retrieval, and NVIDIA's AI models for processing and summarizing images in the documents.

## Prerequisites
First, clone the repo
```bash
git clone https://github.com/DNH35/salesteq-doc-search.git
cd salesteq-doc-search
```
Create a new Python environment using Conda:
```bash
conda create -n doc-search python=3.10
```
Before running the application, ensure you install all the dependencies by running
  ```bash
  pip install -r requirements.txt
   ```

### Setting up your API_KEY
To set up your api key, create an .env file at the project root and set 
```python
API_KEY=<your_api_key_here>
QDRANT_API_KEY=<your_api_key_here>
QDRANT_HOST=<your_host_link_here>
 ```

### Running the script
```bash
python salesteq_doc_search.py input-directory
```

In order to run lc_agent_doc_search.py, copy and paste the file into the lc-agent repo and run it from there