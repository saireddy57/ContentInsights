# from typing import Annotated
import os
from fastapi import FastAPI, File, UploadFile
from utils import utils
import uvicorn
import json
import yt_dlp
import torch
from langchain.docstore.document import Document
from config import Config
import numpy as np

from PIL import Image
from io import BytesIO
import cv2
from paddleocr import PaddleOCR
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv,find_dotenv
import openai


load_dotenv()
app = FastAPI()
whisper_model = Config.whisper_model

doc_path = 'doc/'
chunk_size=3000
chunk_overlap =0
index_name = "pdfchat-app"
ffmpeg_path = "/snap/bin/ffmpeg"  # Replace this with the actual path to ffmpeg if not in PATH



# @app.post("/files/")
# async def create_file(file: Annotated[bytes, File()]):
#     return {"file_size": len(file)}
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
# from langchain.llms import LLM
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

def runner(is_video_img_content,doc_obj = None):
    if not is_video_img_content:
        if '.pdf' in os.listdir(doc_path)[0]:
            document_obj = utils.load_document(doc_path,'.pdf')
        elif ('.doc' in os.listdir(doc_path)[0]) or ('.docx' in os.listdir(doc_path)[0]) :
            document_obj = utils.load_document(doc_path,'.doc')
    else:
        document_obj = doc_obj
    chunked_docs = utils.chunk_data(document_obj,chunk_size,chunk_overlap)
    utils.create_index(index_name)
    global vector_index
    vector_index = utils.prepare_data(chunked_docs)
    # retrival(vector_index)
    return vector_index

def retrival(vector_index):
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(temperature=0.7),#Config.model,model="gpt-3.5-turbo"
        chain_type="map_reduce", 
        retriever=vector_index.as_retriever(),
    )
    admin_prompt = "Following is a question, answer it only based on the document. Here is the question: "#. If answer is not available, respond with 'Answer not found in document'. Do not generate any response other than based on document
    question = admin_prompt + "How much the agriculture target will be increased by how many crore?"
    answer = qa.run(question)
    return qa


@app.post("/get_result")
async def get_res(query):
    # res = vector_index.similarity_search("How much the agriculture target will be increased by how many crore?",k=3)
    

    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-3.5-turbo",temperature=0.9),#Config.model,model="gpt-3.5-turbo"
        chain_type="map_reduce", 
        retriever=vector_index.as_retriever(),
    )
    admin_prompt = "Following is a question, answer it only based on the document. Here is the question: "#. If answer is not available, respond with 'Answer not found in document'. Do not generate any response other than based on document
    question = admin_prompt + query
    answer = qa.run(question)
    
    res = answer#vector_index.similarity_search(query,k=3)

    result = {'result':res}
    print("Result",result)
    # return json.dump(a)
    return result

@app.post("/uploadfiles/")
async def create_upload_file(file: UploadFile):
    if file.filename.split('.')[-1] not in ['png','jpg','jpeg']:
        file_content = await file.read()
        file_path = os.path.join(doc_path,file.filename)
        prev_files = os.listdir(doc_path)
        for prev_file in prev_files:
            os.remove(os.path.join(doc_path,prev_file))
        with open(file_path,'wb') as file_obj:
            file_obj.write(file_content)
        vect_obj = runner(is_video_img_content=False,doc_obj = None)
    elif file.filename.split('.')[-1] in ['png','jpg','jpeg']:
        img_read = await file.read()
        image_str = Image.open(BytesIO(img_read))
        image_array = np.array(image_str)
        ocr_model = PaddleOCR(lang='en')
        result = ocr_model.ocr(image_array)
        content=[]
        for res in result:
            for i in res:
                content.append(i[1][0])
        paragraph = '\n'.join(content)
        # print(paragraph)
        cv2.imwrite(f'{file.filename}',image_array)
        doc = [Document(paragraph)]
        vect_obj = runner(is_video_img_content=True,doc_obj = doc)
        
    return {"status": "Success"}    

# def process_img_data(image_array):


def download_audio(youtube_url, ffmpeg_path=''):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': '%(title)s.%(ext)s',
    }
    
    if ffmpeg_path:
        ydl_opts['ffmpeg_location'] = ffmpeg_path

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(youtube_url, download=True)
        audio_file_path = ydl.prepare_filename(info_dict).replace('.webm', '.mp3')
    return audio_file_path

def extract_text_from_audio(audio_file_path):
    if 'm4a' in audio_file_path:
        audio_file_path = audio_file_path.replace('m4a','mp3')
    result = whisper_model.transcribe(audio_file_path)
    doc_obj = [Document(result['text'])]
    return doc_obj

@app.post("/process_video")
async def process_ytb_video(yt_url):
    audio_file_path = download_audio(yt_url, ffmpeg_path)
    docobj = extract_text_from_audio(audio_file_path)
    runner(is_video_img_content=True,doc_obj=docobj)
    return {"status": "Success"} 
    
# if __name__ == "__main__":
#     uvicorn.run(app, host="127.0.0.1", port=8001)