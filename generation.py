# generation.py
import os
from dotenv import load_dotenv
from google import genai
from augumentation import create_augmented_prompt  # note spelling: augumentation.py
from retrieval import retrieval

load_dotenv()

def generate_response(query: str, max_docs: int = 5) -> str:
    """
    Retrieve -> augment -> generate pipeline.
    - query: user question
    - max_docs: how many retrieved documents to include in context
    """
    retrieved = retrieval(query)  # expects list-of-lists
    if not retrieved:
        return "No relevant information found in the database."

    prompt = create_augmented_prompt(query, retrieved, max_docs=max_docs)

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"Generation failed: {e}"

if __name__ == "__main__":
    sample_query = "Tripods best for DSLR camera"
    print(generate_response(sample_query))
