import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import bot_template , user_template,css
from langchain.llms import HuggingFaceHub

load_dotenv()
# Get the Hugging Face API token from the environment variable
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not hf_token:
    raise ValueError("Hugging Face API token not found. Please check your .env file.")



def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
           text +=  page.extract_text() or ""
    return text     

# def get_docx_text(docx_docs):
#     doc = Document(docx_docs)
#     text = "\n".join([p.text for p in doc.paragraphs])
#     return text  

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        #  separator = ["\n"],
    chunk_size = 1000,
    chunk_overlap = 200,
    length_function = len
    )
    chunks = text_splitter.split_text(text) 
    return chunks

def get_vectorstore(text_chunks):
    # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl" )
    vectorstore = FAISS.from_texts( text_chunks , embedding = embeddings)
    return vectorstore
    
    
def get_conversation_chain(vectorstore):
    # llm = ChatOpenAI()
    llm = HuggingFaceHub(repo_id = "google/flan-t5-xxl", model_kwargs = {"temperature": 0.5 , "max_length": 512 ,"huggingfacehub_api_token": hf_token })
    memory = ConversationBufferMemory(memory_key = 'chat-history' , return_messages = True)
    convesation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vectorstore.as_retriever(),
        memory = memory
    )
    return convesation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation(user_question)
    st.session_state.chat_history = response['chat-history']
    
    for i , message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}" , message.content) , unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}" , message.content) , unsafe_allow_html=True)
   
    
            

def main():
    load_dotenv()
    st.set_page_config(page_title= "Chat with multiple PDF's" , page_icon = ":books:")
    
    
    
    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.header("Chat with multiple PDF's :books:")
    # st.text_input("Ask a question about your documents:")
    
    user_question = st.text_input("Ask a question about your documents:")
    if user_question and st.session_state.conversation:
        handle_userinput(user_question)
    
    vectorstore = None
    
    st.write(user_template.replace("{{MSG}}" , "Hello robot") , unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}" , "Hello Human") , unsafe_allow_html=True)
    
    # with is used to group the elements together
    with st.sidebar:
        st.header("Your documentss")
        pdf_docs = st.file_uploader(
            "Upload your PDF's" , accept_multiple_files=True)
        if st.button("Process it!"):
           with st.spinner("Processing..."):
            #  get pdf text => we will get single string with all the text
            raw_text = get_pdf_text(pdf_docs)
            # st.write(raw_text)
            
            # get the chunk text
            text_chunks = get_text_chunks(raw_text)
            st.write(text_chunks)
            
            
            # create vector store
            vectorstore = get_vectorstore(text_chunks)
        
        
        # create conversation chain
    if vectorstore  and not st.session_state.conversation:
        st.session_state.conversation = get_conversation_chain(vectorstore)
        # it takes the history of the conversation and gives the next element of the conversation

# Only initialize conversation if vectorstore is created
    if st.session_state.conversation:          
       st.session_state.conversation
    
    
    
if __name__ == "__main__":
    main()        