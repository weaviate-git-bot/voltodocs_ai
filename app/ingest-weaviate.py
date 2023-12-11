import logging

import click

from langchain.embeddings import GPT4AllEmbeddings
from langchain.vectorstores import Weaviate
import torch
from .utils import load_documents, split_documents

from constants import (
    DEVICE_TYPES,
    SOURCE_DIRECTORY,
    WEAVIATE_URL,
)


@click.command()
@click.option(
    "--device_type",
    default="cuda" if torch.cuda.is_available() else "cpu",
    type=click.Choice(
        DEVICE_TYPES,
    ),
    help="Device to run on. (Default is cuda)",
)
def main(device_type):
    # Load documents and split in chunks
    logging.info(f"Loading documents from {SOURCE_DIRECTORY}")

    documents = load_documents(SOURCE_DIRECTORY)
    texts = split_documents(documents)

    logging.info(f"Loaded {len(documents)} documents from {SOURCE_DIRECTORY}")
    logging.info(f"Split into {len(texts)} chunks of text")

    embeddings = GPT4AllEmbeddings(client=None)

    if texts:
        db = Weaviate.from_documents(
            texts,
            weaviate_url=WEAVIATE_URL,
            embedding=embeddings,
            index_name="LangChain",
            # text_key="text",
        )
        print(f"Indexed in index: {db._index_name}")

        query = "What is Volto?"
        docs = db.similarity_search(query)
        assert len(docs) > 0
    else:
        logging.warning("No documents found to be indexed")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s",
        level=logging.INFO,
    )
    main()
