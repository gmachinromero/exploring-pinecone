import os
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

if __name__ == "__main__":

    print("Reding documents...")
    loader = TextLoader("medium-blogs/medium-blog-1.txt")
    document = loader.load()

    print("Splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    print("Embedding...")
    embedding = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

    print("Ingesting...")
    PineconeVectorStore.from_documents(
        texts,
        embedding,
        index_name=os.environ.get("INDEX_NAME")
    )

    print("Finished ingestion...")
