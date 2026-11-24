import streamlit as st
from transformers import AutoModelForSeq2SeqLM, PreTrainedTokenizerFast
import torch

# Load the model and tokenizer from HuggingFace model hub
@st.cache_resource
def load_model():
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained("satyat298/llm_headline_gen", weights_only=True)
        tokenizer = PreTrainedTokenizerFast(tokenizer_file="./tokenizer.json")
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
        return None, None

model, tokenizer = load_model()

if model is not None and tokenizer is not None:
    # Function to generate a headline from the input news article
    def generate_headline(text):
        inputs = tokenizer(text, return_tensors="pt", max_length=256, truncation=True, padding="max_length")
        outputs = model.generate(inputs["input_ids"], max_length=64, num_beams=4, repetition_penalty=2.0, length_penalty=1.0)
        headline = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return headline

    # Streamlit UI components
    st.title("News Article Headline Generatorr")
    st.markdown("Enter a news article below, and click 'Generate Headline' to get a summary.")
    
    article = st.text_area("News Article", height=200)
    if st.button("Generate Headline"):
        if article:
            headline = generate_headline(article)
            st.subheader("Generated Headline")
            st.write(headline)
        else:
            st.error("Please enter a news article.")
else:
    st.error("Failed to load model and tokenizer.")