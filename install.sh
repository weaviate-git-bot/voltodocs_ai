.venv/bin/pip install -U pip
.venv/bin/pip install sentence_transformers
.venv/bin/pip install  torch  --index-url https://download.pytorch.org/whl/cu118
.venv/bin/pip install InstructorEmbedding
.venv/bin/pip install langserve
.venv/bin/pip install weaviate-client
.venv/bin/pip install -e packages/voltodocs/
virtualenv .venv
pip install virtualenv
.venv/bin/python app/server.py
.venv/bin/pip install -U langchain-community
.venv/bin/pip install -U faiss-gpu
