from langchain.document_loaders import PyPDFDirectoryLoader,UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
import os 
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import OllamaEmbeddings
import requests
# from langchain.vectorstores import Pinecone 
from config import Config
# from openai import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv,find_dotenv
load_dotenv()
os.environ['PINECONE_API_KEY'] = os.environ.get('PINECONE_API_KEY')
device = torch.device('cpu')

# _ = load_dotenv(find_dotenv())
# embedding = OllamaEmbeddings()
# vectorstore = PineconeVectorStore(index_name="pdfchat-app", embedding=Config.model)
pc = Pinecone(api_key=os.environ.get('PINECONE_API_KEY'))
dimension = 768 #1536 #768
# model = "llama2"
index_name = "pdfchat-app"

# def load_ytb_video():

def load_document(doc_path,doc_type=None):
    if doc_type ==".pdf":
        loader = PyPDFDirectoryLoader(doc_path)
    else:
        file_path = os.path.join(doc_path,os.listdir(doc_path)[0])
        loader = UnstructuredWordDocumentLoader(file_path,mode='single')
        
    doc_obj = loader.load()
    return doc_obj

def chunk_data(docs,chunk_size,chunk_overlap):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    doc=text_splitter.split_documents(docs)
    return doc


def create_index(index_name):
    if pc.list_indexes().names():
        pc.delete_index(pc.list_indexes().names()[0])
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name = index_name ,
            dimension = dimension ,
            metric = "cosine" ,
            spec = ServerlessSpec(
                    cloud='aws' ,
                    region='us-east-1'
            )
        )
        index = pc.Index(index_name)

def generate_embeddings(model,docs):
    res = requests.post(url='http://localhost:11434/api/embeddings',
                        json={
                            'model': model, #'llama2',
                            'prompt': docs#'Hello World How Are You I am Sai'
                        })
    if res.status_code == 200:
        # print(res.json().get('embedding'))
        return res.json().get('embedding')
    else:
        print("Error:", res.status_code)

from langchain import vectorstores


def prepare_data(doc_obj):
    model = Config.model #OpenAIEmbeddings() #Config.model
    vectorstore_from_docs = vectorstores.Pinecone.from_documents(doc_obj,model,index_name=index_name)
    print(vectorstore_from_docs)
    # Implemented for Lama2 API for generating embeddings
    # vectors_to_upsert = []
    # index = pc.Index("pdfchat-app")
    # for i, doc in enumerate(doc_obj):
    #     embedding = generate_embeddings(model,doc.page_content)
    #     vectors_to_upsert.append({
    #             "id": str(i),
    #             "values": embedding,
    #             "metadata": {"text": doc.page_content}
    #             })
    # f = open('outputlist.txt', 'w')
    # simplejson.dump(vectors_to_upsert, f)
    # f.close()
    # index.upsert(vectors=vectors_to_upsert)
    # query_vector = generate_embeddings('llama2',"Hi How Many years of experience did Sai Prabhu Reddy have?")

    # matching_results=vectorstore_from_docs.similarity_search("How much the agriculture target?",k=3)
    return vectorstore_from_docs


