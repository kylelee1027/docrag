from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import glob
import os

folder_path = "data"
pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))

# Load PDF files from data/
loaded_files = []
for file in pdf_files:
    loader = PyPDFLoader(file)
    pages = loader.load()
    loaded_files.append(pages)
    
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

# Splitting loaded documents
split_documents = []
for file in loaded_files:
    split = splitter.split_documents(file)
    split_documents.append(split)

# Init embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Embed and store
for file in split_documents:
    vectorstore = FAISS.from_documents(file, embeddings)
    vectorstore.save_local("vectorstore/")
    