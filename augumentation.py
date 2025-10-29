# from google import genai #Gemini/GenAI client library
# import os
# from dotenv import load_dotenv #Loads variables from a .env file
# from google.genai import types #an access GEMINI_API_KEY 
# from retrieval import retrieval

# load_dotenv()
# def chat(prompt:str):
#     client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
#     response = client.models.generate_content(
#     model = "gemini-2.5-flash-lite", 
#     contents=prompt,
#     )
#     return response

# def prompt_creation_and_api_calls():
#     query = "5 Cheapest tripod to buy"
#     #this gets the retrived products(combined and cleaned) from the query and creates the prompt
#     augmented_document = retrieval(query)
#     prompt = f"""
#     You are a helpful and friendly assistant that answers questions about Daraz products.
#     Use the information from the passages below to provide a clear and complete answer.
#     Explain things in simple terms for a non-technical customer.
#     QUESTION: '{query}'
#     PASSAGE: '{augmented_document}' 
#     """

#     #API CALL
#     response = chat(prompt)
#     print(response.text)

# augumentation.py
from google import genai  # Gemini client
import os
from dotenv import load_dotenv
from retrieval import retrieval

load_dotenv()

def chat(prompt: str):
    """Simple wrapper that calls Gemini using GEMINI_API_KEY from .env"""
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=prompt,
    )
    return response

def create_augmented_prompt(query: str, retrieved_chunks: list, max_docs: int = 5):
    """
    Build a clear, contextual prompt for the model.

    - retrieved_chunks: output of retrieval(), i.e. a list of lists of chunk strings:
        e.g. [['chunk a1','chunk a2'], ['chunk b1','chunk b2']]
    - max_docs: max number of documents to include (keeps prompt size manageable)
    """
    # Flatten each document's chunks into a single document string
    docs = [" ".join(chunks) for chunks in retrieved_chunks][:max_docs]

    # Build readable context sections
    context_sections = []
    for i, doc in enumerate(docs, start=1):
        # You can add more metadata here (doc ids, source) if available
        context_sections.append(f"Document {i}:\n{doc.strip()}")

    context = "\n\n---\n\n".join(context_sections)

    prompt = f"""
You are a helpful and friendly assistant that answers questions about Daraz products.
Use ONLY the information in the 'Context' below to answer the user's question. If the answer cannot be found in the context, say you don't know and avoid inventing facts.
Explain things in simple terms for a non-technical customer.

QUESTION:
{query}

CONTEXT:
{context}

Answer:"""
    return prompt

def prompt_creation_and_api_calls():
    """
    Example flow that uses retrieval(), create_augmented_prompt(), and chat().
    """
    query = "5 Cheapest tripod to buy"
    retrieved = retrieval(query)  # returns list-of-lists (chunks per doc)
    if not retrieved:
        print("No retrieved documents.")
        return

    prompt = create_augmented_prompt(query, retrieved, max_docs=5)
    response = chat(prompt)
    print(response.text)
