import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")
model.load_state_dict(torch.load('best_model.pth'),map_location=torch.device('cpu'))
tokenizer = AutoTokenizer.from_pretrained("google/mt5-small")

# Streamlit app
st.title('Headline Generation System')

# Input text
article_text = st.text_area("Enter the news article:", height=300)

if st.button('Generate Headline'):
    # Tokenize and generate headline
    inputs = tokenizer.encode("summarize: " + article_text, return_tensors="pt")
    summary_ids = model.generate(inputs, max_length=50, min_length=5, length_penalty=2.0, num_beams=4, early_stopping=True)
    headline = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    st.write('Generated Headline:')
    st.write(headline)