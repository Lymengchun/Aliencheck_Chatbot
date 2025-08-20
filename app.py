from fastapi import FastAPI, Form, UploadFile, File
from pydantic import BaseModel
import ollama
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaLLM,OllamaEmbeddings
import utils
import os
import uuid

app = FastAPI()
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="pdf_docs")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")  # Or other Ollama embedding model
UPLOAD_DIR = "uploads"


class Question(BaseModel):
    question: str

class Document(BaseModel):
    title: str


def store_chromadb(docs):
    for i, chunk in enumerate(docs):
        vector = embeddings.embed_query(chunk)  # list[float]
        collection.add(
            embeddings=[vector],
            documents=[chunk],
            ids=[str(i)]
        )

    print(f"Stored {len(docs)} chunks in ChromaDB.")

#Simple GET endpoint
@app.get("/")
def read_root():
    return {"message": "Hello World!"}

# POST endpoint
@app.post("/ask")
def ask_bot(data: Question):
    print("question:",data.question)
    # 1. Embed question
    q_vector = embeddings.embed_query(data.question)

    # 2. Search in ChromaDB
    results = collection.query(
        query_embeddings=[q_vector]
    )

    # 3. Combine retrieved chunks
    context = "\n".join(results["documents"][0])

    # 5. Ask Ollama with context
    # llm = Ollama(model="llama3.2")  # Or other model
    prompt = f"You are aliencheckbot and a question-answering assistant. Answer briefly,short and accurately. If you don't know the answer, say 'I don't know.' using the following context:\n\n{context}\n\nQuestion: {data.question}"
    # answer = llm(prompt)
    response = ollama.chat(
        model='llama3.2',
        messages=[
            {'role': 'user', 'content': prompt}
        ],
        keep_alive=-1
    )

    msgs = [response['message']['content'][i:i + 4096] for i in range(0, len(response['message']['content']), 4096)]
    final_result = ""
    for msg in msgs:
        final_result += msg + "\n"
    print("answer:",final_result)
    return {"answer":final_result}


@app.post("/uploadpdf/")
async def upload_file_with_data(
    file: UploadFile = File(...)
):
        # Ensure upload directory exists
    os.makedirs(UPLOAD_DIR, exist_ok=True)
        # Save file to disk
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    text = utils.read_folder(file_path)
    if not text:  
        return {
        "Error": "Can't read this pdf or PDF is empty or just spaces!!"
        }
    if not text.strip():
        return {
        "Error":"Can't read this pdf or PDF is empty or just spaces!!"
        }

    #Split into smaller chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_text(text)

    embeddings = OllamaEmbeddings(model="mxbai-embed-large")  # Or other Ollama embedding model
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="pdf_docs")

    #Store in ChromaDB
    vectors = [embeddings.embed_query(chunk) for chunk in docs]

    ids = [str(uuid.uuid1().hex[:8]) for i in range(len(docs))]
    metadatas = [
        {"title": file.filename, "chunk_index": i}
        for i in range(len(docs))
    ]

    collection.add(
        embeddings=vectors,
        documents=docs,
        metadatas=metadatas,
        ids=ids,
    )


    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size_mb": round(file.size/(1024*1024),2)
    }



@app.post("/delete_doc/")
def delete_doc(document:Document):
    data = collection.get(where={"title":document.title})
    if not data["ids"]:
        return {"Error":"No documents found with that title."}
    collection.delete(
        where = {"title":document.title}
    )
    
    return {"msg":document.title+" was delete."}

@app.get("/all_docs")
def get_docs():
    data = collection.get()
    if not data:
        return {"docs":[]}
    titles = [m["title"] for m in data["metadatas"] if "title" in m]
    unique_titles = list(set(titles))
    return {"docs":unique_titles}
    