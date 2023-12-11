import weaviate

from langchain.retrievers.weaviate_hybrid_search import WeaviateHybridSearchRetriever
from langchain.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings


class Question(BaseModel):
    __root__: str


def weaviate_retriever(constants):
    client = weaviate.Client(
        url=constants.WEAVIATE_URL,
    )
    retriever = WeaviateHybridSearchRetriever(
        client=client,
        index_name="LangChain",
        text_key="text",
        attributes=[],
        create_schema_if_missing=True,
    )
    return retriever


def faiss_retriever(EMBEDDING_MODEL_NAME):
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cuda"},
    )
    with open("DB/faiss.pickke", "rb") as f:
        bindata = f.read()

    db = FAISS.deserialize_from_bytes(bindata, embeddings)
    return db.as_retriever()


def make_chain(
    constants,
):
    retriever = faiss_retriever(constants.EMBEDDING_MODEL_NAME)
    # Optionally, pull from the Hub
    # from langchain import hub
    # prompt = hub.pull("rlm/rag-prompt")
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    ollama_llm = "llama2:7b-chat"
    model = ChatOllama(model=ollama_llm)

    # RAG chain
    chain = (
        RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()})
        | prompt
        | model
        | StrOutputParser()
    )

    chain = chain.with_types(input_type=Question)
    return chain
