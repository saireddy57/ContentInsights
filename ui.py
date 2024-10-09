import streamlit as st
import requests 
from config import Config

# allowed_file_types
img_formats = ['png','jpg','jpeg']
img_repr = ['image/png','image/jpeg','image/jpeg']

def read_pdf(uploaded_file):
    # uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file:
        file_data = uploaded_file.getvalue()
        if '.pdf' in uploaded_file.name:
            files = [("file", (uploaded_file.name, file_data, "application/pdf"))]
        elif '.docx' in uploaded_file.name:
            mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            files = [("file", (uploaded_file.name, file_data, mime_type))]
        elif uploaded_file.name.split('.')[-1] in img_formats:
            img_format_repr = img_repr[img_formats.index(uploaded_file.name.split('.')[-1])]
            files = [("file", (uploaded_file.name, file_data, img_format_repr))]

            # img_format_repr = img_repr[img_formats.index(uploaded_file.name.split('.')[-1])]
        url = "http://127.0.0.1:8000/uploadfiles/"
        response = requests.request("POST",url=url,files=files)
        # st.write(uploaded_file.getvalue())
        return eval(response.text)["status"]

def process_ytb_video(yt_url):
    st.write(yt_url)
    url = "http://127.0.0.1:8000/process_video/?yt_url={0}".format(yt_url)
    response = requests.post(url=url)
    return eval(response.text)["status"]


def get_response(query):
    st.write("Hi You clicked the button")
    url = "http://127.0.0.1:8000/get_result/?query={0}".format(query)
    response = requests.post(url=url)
    response_text = response.text
    # if 'null' in response_text:
    #     response_text = response_text.replace("null","")
        
    return response_text

# Initialize session state
if 'file_uploaded' not in st.session_state:
    st.session_state['file_uploaded'] = False
if 'url_accepted' not in st.session_state:
    st.session_state['url_accepted'] = False
if 'response' not in st.session_state:
    st.session_state['response'] = None
if 'query' not in st.session_state:
    st.session_state['query'] = ""
if 'uploaded_file_content' not in st.session_state:
    st.session_state['uploaded_file_content'] = None

# Function to handle file upload
def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1]
    if file_extension in Config.allowed_file_types:
        with st.spinner('Wait for setting up the VectorDB'):
            response = read_pdf(uploaded_file)
        if response == "Success":
            st.session_state['file_uploaded'] = True
            st.session_state['response'] = response
            st.session_state['uploaded_file_content'] = uploaded_file.getvalue()
            st.success("File processed successfully!")
        else:
            st.error("Failed to process the file.")
    else:
        st.error("Invalid file type.")

# Radio button for file type selection
st.header('Select Source', divider='rainbow')
genre = st.radio("Select an Option", ["", "Text Files","Image", "Video"])

if (genre == "Text Files") or (genre == "Image"):
    st.write("You have selected Text Files")
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file:
        # Check if the file is new or different from the previously uploaded file
        if st.session_state['uploaded_file_content'] != uploaded_file.getvalue():
            handle_file_upload(uploaded_file)
    else:
        st.write("No input given")
elif genre == "Video":
    url_link = st.text_input("Please paste youtube url here....")
    if st.button("Process Video"):
        response = process_ytb_video(url_link)
        if response == "Success":
            st.session_state['url_accepted'] = True
            st.session_state['response'] = "Success"


# Check if file was successfully uploaded and processed
if (st.session_state['file_uploaded'] or st.session_state['url_accepted']) and st.session_state['response'] == "Success":
    st.write("File has been processed successfully!")
    query = st.text_input("Enter the Query", value=st.session_state['query'])
    st.session_state['query'] = query
    if st.button("Call"):
        st.write(f"Query: {query}")
        with st.spinner('Wait until the query is hit...'):
            res = get_response(query)
        st.write("Query result:", res)
else:
    st.write("Please upload and process a file first.")

