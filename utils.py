import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# def read_folder(folder_path:str):
#     text = ""
#     # doc = []
#     for filename in os.listdir(folder_path):
#         if filename.lower().endswith(".pdf"):
#             pdf_path = os.path.join(folder_path, filename)
#             print("reading ",pdf_path)
#             reader = PyPDFLoader(pdf_path)
        
#             pages = reader.load()
#             # doc.extend(reader.load())
#             for page in pages:
#                 text += page.page_content + "\n"

#     return text

def read_folder(file_file:str):
    text = ""
    reader = PyPDFLoader(file_file)
    pages = reader.load()
    for page in pages:
        text += page.page_content + "\n"

    return text

def split_chunks(text):
    # 2. Split into smaller chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_text(text)
    return docs