import streamlit as st
import os
import json

def memory_calculator(model_size, gpus, hidden_size, vocab, layers, total_sequence):
    total_model_memory = model_size * 18 * 1000000000 / (gpus * 1024*1024*1024)
    activation_checkpoints = 2 * layers * hidden_size * total_sequence/ (gpus * 1024*1024*1024)
    activation_working_memory = 40 * hidden_size * total_sequence / (gpus * 1024 * 1024 * 1024)
    logits_working_memory = 4 * total_sequence * vocab / (gpus * 1024 * 1024 * 1024) 
    total_memory = total_model_memory + activation_checkpoints + activation_working_memory + logits_working_memory
    return total_memory

# Load JSON config options from local models/ directory
MODEL_DIR = "models"
model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".json")]
model_names = [os.path.splitext(f)[0] for f in model_files]

st.title("Model Training Memory Calculator")

with st.form("memory_form"):
    model_choice = st.selectbox("Select a model config", model_names)

    # Load selected model JSON
    model_path = os.path.join(MODEL_DIR, f"{model_choice}.json")
    with open(model_path, "r") as f:
        config = json.load(f)
    
    hidden_size = config.get("hidden_size", 4096)
    vocab = config.get("vocab_size", 50000)
    layers = config.get("num_layers", 32)

    st.write(f"📄 Loaded Config: `hidden_size={hidden_size}`, `vocab_size={vocab}`, `layers={layers}`")

    model_size = st.number_input("Model Size (in Billions, e.g. 7 for 7B)", min_value=0.1, value=7.0, step=0.1)
    gpus = st.number_input("Number of GPUs", min_value=1, value=8)
    total_sequence = st.number_input("Total Sequence Length (e.g. 2048)", min_value=1, value=2048)
    
    submitted = st.form_submit_button("Calculate Memory")
    if submitted:
        total_memory = memory_calculator(model_size, gpus, hidden_size, vocab, layers, total_sequence)
        st.success(f"Estimated Memory per GPU: {total_memory:.2f} GB")
