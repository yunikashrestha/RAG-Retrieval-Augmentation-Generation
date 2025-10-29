# app.py
import streamlit as st
from generation import generate_response

st.set_page_config(page_title="Daraz Product Assistant", page_icon="🛍️", layout="centered")

st.title("Daraz Product AI Assistant")
st.markdown(
    """
    Type your product-related question below, and the assistant will retrieve relevant 
    information from the database and generate an AI-powered answer using Gemini.
    """
)

# User input
query = st.text_input("🔎 Enter your query:", placeholder="e.g., 5 Cheapest tripods to buy")

# Button to generate answer
if st.button("Generate Answer"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        with st.spinner("🔍 Retrieving and generating response..."):
            try:
                answer = generate_response(query)
                st.success("Generated Response:")
                st.write(answer)
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Footer
st.markdown(
    """
    ---
    **Powered by Gemini + FastEmbed + Qdrant**
    """
)
