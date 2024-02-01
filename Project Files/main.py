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
