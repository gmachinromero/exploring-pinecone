import os

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_text_splitters import CharacterTextSplitter


load_dotenv()




if __name__ == "__main__":
    print("Hi")

    file_path = "./pdf-files/react-paper.pdf"
    loader = PyPDFLoader(file_path=file_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=30,
        separator="\n"
    )

    docs = text_splitter.split_documents(documents=documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    print(documents)