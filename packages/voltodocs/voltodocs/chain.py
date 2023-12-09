import weaviate
from langchain.vectorstores import Weaviate
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chat_models import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough

# from langchain.embeddings import OllamaEmbeddings
# from langchain.document_loaders import WebBaseLoader
# from langchain.embeddings import GPT4AllEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import Chroma


# Add typing for input
class Question(BaseModel):
    __root__: str


def make_chain(
    WEAVIATE_URL,
    EMBEDDING_MODEL_NAME,
):
    embeddings = HuggingFaceInstructEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cuda"},
    )
    client = weaviate.Client(
        url=WEAVIATE_URL,
    )
    db = Weaviate(
        client=client,
        text_key="text",
        index_name="voltodocs",
    )
    retriever = db.as_retriever()

    # Prompt
    # Optionally, pull from the Hub
    # from langchain import hub
    # prompt = hub.pull("rlm/rag-prompt")
    # Or, define your own:
    template = """Answer the question based only on the following context:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    # LLM
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
