from voltodocs import make_chain

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes

from constants import (
    WEAVIATE_URL,
    EMBEDDING_MODEL_NAME,
    # DOCUMENT_MAP,
    # INGEST_THREADS,
    # PERSIST_DIRECTORY,
    # SOURCE_DIRECTORY,
)


app = FastAPI()
voltodocs_chain = make_chain(
    WEAVIATE_URL,
    EMBEDDING_MODEL_NAME,
    # DOCUMENT_MAP,
    # INGEST_THREADS,
    # PERSIST_DIRECTORY,
    # SOURCE_DIRECTORY,
)


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


add_routes(app, voltodocs_chain, path="/voltodocs")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
