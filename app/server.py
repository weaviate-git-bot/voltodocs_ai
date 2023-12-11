from voltodocs import make_chain

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes

import constants


app = FastAPI()
voltodocs_chain = make_chain(constants)


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


add_routes(app, voltodocs_chain, path="/voltodocs")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
