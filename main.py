import os

from dotenv import load_dotenv
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore

load_dotenv()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


if __name__ == "__main__":

    query = "What is Pinecone in machine learning?"

    print("Retrieving from LLM...")
    embeddings = OpenAIEmbeddings()
    llm = ChatOpenAI()
    chain = PromptTemplate.from_template(template=query) | llm
    result = chain.invoke(input={})
    print(result.content)


    print("Retrieving from RAG v1...")
    vectorstore = PineconeVectorStore(
        index_name=os.environ.get("INDEX_NAME"),
        embedding=embeddings
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
        llm,
        retrieval_qa_chat_prompt
    )

    retrieval_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(),
        combine_docs_chain=combine_docs_chain
    )

    result = retrieval_chain.invoke(input={"input": query})
    print(result)


    print("Retrieving from RAG v2...")
    template = """Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an aswer.
    Use three sentences maximum and keep the answer as concise as possible.
    Always say "thanks for asking!" at the end of the answer.
    
    {context}
    
    Question: {question}
    
    Helpful Answer:
    """

    custom_rag_prompt = PromptTemplate.from_template(template)

    rag_chain = (
        {"context": vectorstore.as_retriever() | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
    )

    result = rag_chain.invoke(query)

    print(result)
