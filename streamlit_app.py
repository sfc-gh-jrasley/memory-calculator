import streamlit as st

def memory_calculator(model_size, gpus, hidden_size, vocab, layers, total_sequence):
    total_model_memory = model_size * 18 * 1000000000 / (gpus * 1024*1024*1024)
    activation_checkpoints = 2 * layers * hidden_size * total_sequence/ (gpus * 1024*1024*1024)
    activation_working_memory = 40 * hidden_size * total_sequence / (gpus * 1024 * 1024 * 1024)
    logits_working_memory = 4 * total_sequence * vocab / (gpus * 1024 * 1024 * 1024) 
    total_memory = total_model_memory + activation_checkpoints + activation_working_memory + logits_working_memory
    return total_memory

st.title("Model Training Memory Calculator")

with st.form("memory_form"):
    model_size = st.number_input("Model Size (in Billions, e.g. 7 for 7B)", min_value=0.1, value=7.0, step=0.1)
    gpus = st.number_input("Number of GPUs", min_value=1, value=8)
    hidden_size = st.number_input("Hidden Size (e.g. 4096)", min_value=1, value=4096)
    vocab = st.number_input("Vocabulary Size (e.g. 50000)", min_value=1, value=50000)
    layers = st.number_input("Number of Layers (e.g. 32)", min_value=1, value=32)
    total_sequence = st.number_input("Total Sequence Length (e.g. 2048)", min_value=1, value=2048)
    
    submitted = st.form_submit_button("Calculate Memory")
    if submitted:
        total_memory = memory_calculator(model_size, gpus, hidden_size, vocab, layers, total_sequence)
        st.success(f"Estimated Memory per GPU: {total_memory:.2f} GB")
