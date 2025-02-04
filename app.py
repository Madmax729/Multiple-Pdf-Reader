import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from docx import Document

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
           text +=  page.extract_text()
    return text     

# def get_docx_text(docx_docs):
#     doc = Document(docx_docs)
#     text = "\n".join([p.text for p in doc.paragraphs])
#     return text   
            




def main():
    load_dotenv()
    st.set_page_config(page_title= "Chat with multiple PDF's" , page_icon = ":books:")
    
    st.header("Chat with multiple PDF's :books:")
    st.text_input("Ask a question about your documents:")
    
    # with is used to group the elements together
    with st.sidebar:
        st.header("Your documentss")
        pdf_docs = st.file_uploader(
            "Upload your PDF's" , accept_multiple_files=True)
        if st.button("Process it!"):
           with st.spinner("Processing..."):
            #  get pdf text => we will get single string with all the text
            raw_text = get_pdf_text(pdf_docs)
            st.write(raw_text)
            
            # get the chunk text
            
            
            
            # create vector store
               
        
        
        
        
if __name__ == "__main__":
    main()        