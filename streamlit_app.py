import streamlit as st
import os
import json
from huggingface_hub import hf_hub_download

models = ["meta-llama/Llama-3.3-70B-Instruct", "meta-llama/Llama-3.1-8B-Instruct", "Qwen/QwQ-32B", "Qwen/Qwen2.5-14B-Instruct-1M"]

def fetch(filename, model_name):
    file_path = hf_hub_download(repo_id=model_name, filename=filename)
    return json.load(open(file_path))

def fetch_stats(model_name):
    config = fetch("config.json", model_name)
    index = fetch("model.safetensors.index.json", model_name)
    model_size = index["metadata"]["total_size"] // 1e9 // 2
    return config, model_size

def memory_calculator(model_size, gpus, hidden_size, vocab, layers, total_sequence, offload_optim=False):
    total_model_memory = model_size * 18 * 1000000000 / (gpus * 1024*1024*1024)
    if offload_optim:
        optim_states = model_size * 12 * 1000000000 / (gpus * 1024*1024*1024)
        total_model_memory -= optim_states
    activation_checkpoints = 2 * layers * hidden_size * total_sequence/ (gpus * 1024*1024*1024)
    activation_working_memory = 40 * hidden_size * total_sequence / (gpus * 1024 * 1024 * 1024)
    logits_working_memory = 4 * total_sequence * vocab / (gpus * 1024 * 1024 * 1024) 
    total_memory = total_model_memory + activation_checkpoints + activation_working_memory + logits_working_memory
    return total_memory

st.title("Model Training Memory Calculator")

with st.form("memory_form"):
    model_choice = st.selectbox("Select a model config", models)
    st.write(f"{model_choice=}")

    config, model_size = fetch_stats(model_choice)
    
    hidden_size = config.get("hidden_size", -1)
    vocab = config.get("vocab_size", -1)
    layers = config.get("num_hidden_layers", -1)

    st.write(f"📄 Loaded Config: `hidden_size={hidden_size}`, `vocab_size={vocab}`, `layers={layers}`, `model_size={model_size}`")

    offload = st.checkbox("Offload optimizer states to CPU memory", value=True)

    gpus = st.number_input("Number of GPUs", min_value=1, value=8)

    total_sequence = st.number_input("Total Sequence Length (e.g. 2048)", min_value=1, value=2048)
    
    submitted = st.form_submit_button("Calculate Memory")
    if submitted:
        total_memory = memory_calculator(model_size, gpus, hidden_size, vocab, layers, total_sequence, offload)
        st.success(f"Estimated Memory per GPU: {total_memory:.2f} GB")
