import os
from dotenv import load_dotenv

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()


if __name__ == "__main__":
    print("Welcome to the in-memory RAG =)")

    # Load PDF file
    file_path = "./pdf-files/react-paper.pdf"
    loader = PyPDFLoader(file_path=file_path)
    documents = loader.load()

    # Split into chunks
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=30,
        separator="\n"
    )

    docs = text_splitter.split_documents(documents=documents)

    # Instance in-memory vectorstore
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index_react")

    new_vectorstore = FAISS.load_local(
        "faiss_index_react",
        embeddings,
        allow_dangerous_deserialization=True
    )

    retrieval_qa_chat_prompt = hub.pull(
        owner_repo_commit="langchain-ai/retrieval-qa-chat",
        api_url="https://api.smith.langchain.com"  # Change for US or Europe
    )
    """
    SYSTEM
    Answer any use questions based solely on the context below:

    <context>
    {context}
    </context>

    PLACEHOLDER
    chat_history

    HUMAN
    {input}
    """

    combine_docs_chain = create_stuff_documents_chain(
        OpenAI(),
        retrieval_qa_chat_prompt
    )

    retrieval_chain = create_retrieval_chain(
        new_vectorstore.as_retriever(),
        combine_docs_chain
    )

    res = retrieval_chain.invoke({"input": "Give me the gist of react in 3 sentences"})

    print(res.get("answer"))
