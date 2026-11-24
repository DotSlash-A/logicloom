import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings('ignore')

@st.cache_resource
def load_model():
    model = AutoModelForCausalLM.from_pretrained("fine_tuned_smollm_news")
    tokenizer = AutoTokenizer.from_pretrained("fine_tuned_smollm_news")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    return model, tokenizer, device

def generate_title(article, model, tokenizer, device):
    prompt = f"Article: {article} Title:"
    inputs = tokenizer(
        prompt, 
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    attention_mask = inputs['attention_mask'].to(device)
    inputs = inputs['input_ids'].to(device)
    
    outputs = model.generate(
        inputs,
        attention_mask=attention_mask,
        max_length=len(article),
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        temperature=st.session_state.temperature,
        top_p=st.session_state.top_p,
        top_k=st.session_state.top_k
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

st.title("News Title Generator")

if 'temperature' not in st.session_state:
    st.session_state.temperature = 0.7
if 'top_p' not in st.session_state:
    st.session_state.top_p = 0.9
if 'top_k' not in st.session_state:
    st.session_state.top_k = 50

model, tokenizer, device = load_model()

with st.sidebar:
    st.header("Generation Parameters")
    st.session_state.temperature = st.slider("Temperature", 0.1, 1.0, 0.7)
    st.session_state.top_p = st.slider("Top-p", 0.1, 1.0, 0.9)
    st.session_state.top_k = st.slider("Top-k", 1, 100, 50)

article = st.text_area("Enter news article:", height=200)

if st.button("Generate Title"):
    if article:
        with st.spinner("Generating title..."):
            title = generate_title(article, model, tokenizer, device)
            st.success("Generated Title: " + title.split("Title:")[-1].strip())
    else:
        st.warning("Please enter an article first.")

# uploaded_file = st.file_uploader("Or upload a CSV file with articles", type=['csv'])
# if uploaded_file:
#     import pandas as pd
#     df = pd.read_csv(uploaded_file)
#     if 'News Article' in df.columns:
#         if st.button("Generate Titles for CSV"):
#             progress_bar = st.progress(0)
#             results = []
#             for i, row in df.iterrows():
#                 title = generate_title(row['News Article'], model, tokenizer, device)
#                 results.append(title.split("Title:")[-1].strip())
#                 progress_bar.progress((i + 1) / len(df))
            
#             df['Generated_Title'] = results
#             st.success("Generated titles for all articles!")
#             st.dataframe(df)
            
#             st.download_button(
#                 "Download Results",
#                 df.to_csv(index=False).encode('utf-8'),
#                 "generated_titles.csv",
#                 "text/csv"
#             )
#     else:
#         st.error("CSV must contain 'News Article' column")