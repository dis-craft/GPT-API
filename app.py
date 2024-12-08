import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load GPT model
@st.cache_resource
def load_model():
    model_name = "distilgpt2"  # Lightweight GPT model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

# Initialize the model and tokenizer
tokenizer, model = load_model()

# Function to generate a response
def get_response(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response[len(prompt):].strip()

# Streamlit UI
st.title("Chatbot (GPT Wrapper)")
st.markdown("This is a lightweight GPT wrapper deployed as a chatbot on your website!")

# Input
user_input = st.text_input("You: ", placeholder="Ask me anything!")
if st.button("Send"):
    if user_input:
        bot_response = get_response(user_input)
        st.text_area("Chatbot:", value=bot_response, height=150)
