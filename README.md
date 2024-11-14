# Exploring Pinecone and RAGs

## Instalación

Para clonar este repositorio y comenzar a experimentar con LangChain y Pinecone, sigue estos pasos:

```bash
git clone https://github.com/gmachinromero/exploring-pinecone.git
cd exploring-pinecone
```

Asegúrate de tener instalado Python 3.11+ y pipenv para crear un entorno con todas las dependencias necesarias:

```bash
pipenv install
```

Para ejecutar el RAG que trabaja con un LLM y el vectorstore de Pinecone:
```bash
python pinecone-rag.py
```

Adicionalmente necesitarás los token siguientes en un fichero de configuración `.env`:
- `OPENAI_API_KEY`
- `INDEX_NAME`
- `PINECONE_API_KEY`
- `LANGCHAIN_TRACING_V2`
- `LANGCHAIN_ENDPOINT`
- `LANGCHAIN_API_KEY`
- `LANGCHAIN_PROJECT`


## Recursos
- https://python.langchain.com/v0.2/docs/introduction/
- https://docs.pinecone.io/home
- Udemy: LangChain- Develop LLM powered applications with LangChain

