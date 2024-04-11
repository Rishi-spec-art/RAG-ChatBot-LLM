import streamlit as st
import os
from model import chatModel
import model_utils

FILES_DIR = os.path.normpath(
os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "files")
)

ACCESS_TOKEN = st.secrets["ACCESS_TOKEN"]

st.title("LLM Chatbot RAG Assistant")

@st.cache_resource
def load_model():
    model = chatModel(model_id="google/gemma-2b-it", device = 'cpu', ac_token = ACCESS_TOKEN)
    return model

@st.cache_resource
def load_encoder():
    encoder = model_utils.Encoder(
        model_name="sentence-transformers/all-MiniLM",
        device='cpu'
    )
    return encoder

model = load_model()
encoder = load_encoder()

def save_file(uploaded_file):
    """Helper function to save documents to disk"""
    file_path = os.path.join(FILES_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.get_buffer())
    return file_path


with st.sidebar:
    #inputs and parameters in sidebar
    max_new_tokens = st.number_input("max_new_tokens", 128, 4096, 512)
    k = st.number_input("k", 1, 10, 3)
    uploaded_files = st.file_uploader(
        "Upload PDFs for context", type=['PDF', 'pdf'], accept_multiple_files=True
    )
    file_paths = []
    for uploaded_file in uploaded_files:
        file_paths.append(uploaded_file)
    
    if uploaded_files !=[]:
        #create vector db from retrieved docs
        docs = model_utils.load_and_split_pdfs(file_paths)
        DB = model_utils.FaissDB(docs=docs, embedding_function=encoder.embedding_function)
        

#Initialize chat history        
if "messages" not in st.session_state:
    st.session_state.messages = []

#Display chat messages from hsitory on app rerun
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])
        
        
#Accept user input
if prompt := st.chat_input("Ask me anything!"):
    #Add user message to chat history
    st.session_state.messages.append({'role':'user', 'content': prompt})
    #Display user message in chat message container
    with st.chat_message('user'):
        st.markdown(prompt)
    
    #Display assistant response in chat message container
    with st.chat_message("assistant"):
        user_prompt = st.session_state.messages[-1]['content']
        context = (
            None if uploaded_files==[] else DB.similarity_search(user_prompt, k = k)
        )
        answer = {model.generate(
            user_prompt, context=context, max_new_tokens=max_new_tokens
        )}
        response = st.write(answer)
    st.session_state.messages.append({'role':'assistant', 'content': answer})
