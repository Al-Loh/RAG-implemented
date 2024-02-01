# RAG-implemened

A simple prototype version of RAG process. This goes through all the basic steps of:

1. Getting data (HTML in my case).
2. Processing data.
3. Creating documents or small chunks in other words.
4. Creating word embeddings and storing them in a vector database (ChromaDB).
5. Retrieveing the data and creating a prompt for model (Chatgpt-3.5).
6. Generating response or answer based on context.

# Python dependencies:
```
pip3 install chromadb html5lib pandas openai tiktoken  
```

# HTML web Crawler: For now this is empty will add later

I have yet to complete this. Reason I want the code to take the relevant content and add it in pandas DataFrame.
This way I already have chunks and also I have more control on which data to concatenate together for better results.

```

```

# Data Wrangle: This will be different for everyone.

The code below is very personal it would be much different for everyone however for now i have just kept it simple and wrangled FaQs.
It looks big because it is in pandas I could have used json library from python nvr mind me.

```
import pandas as pd

# ABOVE ARE THE MAIN IMPORTS SPECIFIC TO THIS SCRIPT.


# THE PREPROCESS FUNCTION TAKES IN JSON AND TRANSFORMS IT.

# IT, ALSO DOES ALL THE NECESSARY CLEANING STEPS IF NEEDED.

def wrangle(df_temp=pd.DataFrame()):
    temp = []
    questions = []
    answers = []
    for column in df_temp.columns:
        temp.append(df_temp[column].dropna())
    for series in temp:
        for question in series.index:
            questions.append(question)
    for series in temp:
        for answer in series:
            answers.append(answer)
    data_dict = {"questions": questions, "answers": answers}
    df_2 = pd.DataFrame(data_dict)
    return df_2


# WE READ THE JSON FILE AND SEND TO THE PREPROCESS FUNCTION.

df = pd.read_json("faqs.json")
data = wrangle(df)
data.to_csv('data-sources/processed.csv', index=False)


```

# Setting Up a Vector DataBase: ChromaDB

The following database was choosen for prototyping as it was fairly straight forward and easy to setup.

In the following code the model used for embeddings is not from openai.
The results are really great however and I am going to be changing the DataBase later on so i haven't made much effort,
to write custom embedding function code for chromaDB.

A little docs for the embedding function from https://docs.trychroma.com/embeddings:

"By default, Chroma uses the Sentence Transformers all-MiniLM-L6-v2 model to create embeddings. 
This embedding model can create sentence and document embeddings that can be used for a wide variety of tasks. 
This embedding function runs locally on your machine, and may require you download the model files (this will happen automatically)."

BTW I have not played with distance metric yet right now it is set to cosine similarity. It gives good results however.

```
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


```
# Main Question Answering Generation Script:

The create_context function is mostly the above code plus some string manipulation to create the context.
All it does is retreieve data top_k here would be n_results. That retrieved data is the concatenated into one paragraph.
Top_k = 3 gives better results according to research by AI engineer (https://youtu.be/TRjq7t2Ms5I?si=rTLENCwJvS6NTN3f).

The genrate_answer function is simply taking question inputted by user and context previously retreived on that question.
Then just give out an answer. Note: Do not trust OpenAI docs. (some old code some wrong)

Again will have to play with model parameters like top_p and temprature although these settings again are giving great results.

```
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

API_KEY=""
openai_client = OpenAI(api_key=API_KEY)


def create_context(question: ""):

    # CREATING CLIENT OBJECT OF CHROMA DB BUT OFFLINE MODE.
    chroma_client = chromadb.PersistentClient(path="data-sources/")

    # USING DEFAULT EMBEDDER FOR LATER WILL CONVERT TO GPT TEXT EMBEDDER
    default_ef = embedding_functions.DefaultEmbeddingFunction()

    # IF THERE IS NOT A DATABASE ALREADY THERE WITH THE NAME THEN CREATE ONE.
    collection = chroma_client.get_or_create_collection(name="test-collection", metadata={"hnsw:space": "cosine"},
                                                        embedding_function=default_ef)
    result = collection.query(query_texts=question, n_results=3)
    context = ""
    for doc in result["documents"][0]:
        context = context + doc + "\n"

    return context


def generate_answer(context="January 31st", question="What day is it?", max_tokens=500):
    pass

    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system",
             "content": "Answer the question based on the context below, and if the question can't be answered based "
                        "on the context, say \"I don't know\"\n\n"},
            {"role": "user", "content": f"Context: {context}\n\n---\n\nQuestion: {question}\nAnswer:"}
        ],
        temperature=0.1,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None)

    return response.choices[0].message.content.strip()


# question = input("Prompt: ")
# context = create_context(question)
print(generate_answer(max_tokens=250))

```
