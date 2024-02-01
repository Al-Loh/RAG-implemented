import chromadb
import pandas as pd
from chromadb.utils import embedding_functions

# ABOVE ARE THE MAIN IMPORTS SPECIFIC TO THIS SCRIPT.

# INITIALIZATION OF VARIABLES.
data = pd.read_csv("data-sources/processed.csv")


def vectordb(df: pd.DataFrame):
    # CREATING CLIENT OBJECT OF CHROMA DB BUT OFFLINE MODE.
    client = chromadb.PersistentClient(path="data-sources/")

    # IF THERE IS NOT A DATABASE ALREADY THERE WITH THE NAME THEN CREATE ONE.
    default_ef = embedding_functions.DefaultEmbeddingFunction()
    collection = client.get_or_create_collection(name="test-collection", metadata={"hnsw:space": "cosine"},
                                                 embedding_function=default_ef)

    # ADDING ALL THE INFO IN DATABASE.
    docs = df["answers"].tolist()
    ids = [("id" + str(i)) for i in df.index.to_list()]

    collection.add(documents=docs, ids=ids)


vectordb(data)
