# ContentInsight

ContentInsight processes PDFs, videos, and images to provide accurate, AI-driven question-answering based on the analyzed content.

Tech used:
1. FastAPI - for backend APIs
2. Streamlit - for frontend
3. Langchain - for backend logic
4. GPT-3.5-turbo API calls
5. Nomic-AI-embedding model using HuggingFace
6. Prompt Engineering for different types of AI responses, and also to stop Prompt Injection Attacks
7. Activeloop as vector database
8. Retrieval augmented generation for QA
9. paddleOCR for text in image parsing and OCR

### How to run locally

1. Clone the repository
2. Create a conda environment conda create -n "ContentInsight" python=3.10 and activate it conda activate ContentInsight.
3. Install requirements.txt using the command pip install -r requirements.txt.
4.  Create `.env` file with your own `OPENAI_API_KEY` and `ACTIVELOOP_TOKEN` at the root location of the project. Each can be obtained after creating account on `OpenAI` and    
    `Activeloop`websites.You will be charged by OpenAI. The format of `.env` is as following:
   ```
   OPENAI_API_KEY=YOURKEY
   ACTIVELOOP_TOKEN=YOURKEY
   ```
5. Run **TWO terminals**
   1. FastAPI - backend server **fastapi dev backend.py**
   2. Streamlit - UI **streamlit run ui.py**
6. You can access the application using the URL from the UI server.

Open any browser and go to `127.0.0.1:8501` to see the project

 
### Docs and PDFs Q&A

https://github.com/user-attachments/assets/0d46ac10-8f13-457a-acb3-4111a6fc4ed1

### Youtube videos Q&A

https://github.com/user-attachments/assets/c2b74438-60b2-489b-b489-5d107f2a6af9

### images content Q&A

https://github.com/user-attachments/assets/227203f3-2934-4181-b50c-fef2d18c58d6
