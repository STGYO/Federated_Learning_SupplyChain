# 🥛 Federated Learning Supply Chain 5.0

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?style=for-the-badge&logo=pytorch)
![Hugging Face](https://img.shields.io/badge/Hugging_Face-Models-FFD21E?style=for-the-badge&logo=huggingface)
![Unsloth](https://img.shields.io/badge/Unsloth-Fine_Tuning-9CF?style=for-the-badge)

A cutting-edge, AI-driven **Federated Learning Simulation Dashboard** specifically engineered for the Dairy Supply Chain (Milk).

This project simulates privacy-preserving federated machine learning across three geographically distinct clients (**Amul Gujarat**, **Mother Dairy Delhi**, and **Sudha Bihar**). It actively trains a base Neural Network to optimize profit, reduce waste, and manage carbon caps while utilizing a **custom Fine-Tuned Qwen2.5-0.5B-Instruct LLM** ("SupplyChainGPT") to act as a human-in-the-loop strategist over the network's predictions.

---

## 🚀 Key Features

- **🔒 Federated Learning Simulation:** Three active client nodes train a deep neural network privately on their own data. Gradients are aggregated securely using a central server.
- **🛡️ Differential Privacy & Security:** Built-in gradient clipping, Laplace noise injection, and poison-attack thresholds ensure that no single supply client can disrupt or compromise the network.
- **🤖 Custom LLM Agent (SupplyChainGPT):** A custom Qwen2.5-0.5B-Instruct model fine-tuned specifically on milk supply chain data using **Unsloth 4-bit QLoRA**.
- **⚡ Lightning Fast Offline Inference:** The system loads merged FP16 `.safetensors` with HuggingFace Transformers directly from local storage for fast inference.
- **📊 Beautiful Streamlit UI:** Fully interactive dashboard to observe federated training, trigger simulation rounds, and request AI decision overrides.

---

## 🧠 The AI Pipeline

1. **Dataset Generation:** Raw client supply chain CSVs are run through `generate_finetune_dataset.py` to create over 2,000 ChatML instruction-response pairs.
2. **Cloud Training:** `finetune_qwen_supply_chain.py` uses an NVIDIA T4 GPU on Google Colab to attach LoRA adapters and train supply chain behavior.
3. **Model Centralization:** Unsloth exports LoRA weights, merges into a **16-bit PyTorch model**, and optionally emits **GGUF** formats.

---

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Prithwiraj731/Federated_Learning_SupplyChain.git
cd Federated_Learning_SupplyChain
```

### 2. Set Up Virtual Environment

```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Provide the Fine-Tuned Model Weights

This code relies on custom AI models that are too large to store on GitHub.

1. Make sure you have downloaded the `SupplyChain_Qwen` folder.
2. Place the `SupplyChain_Qwen` folder directly in the project root.

### 5. Launch the Dashboard

```bash
streamlit run app.py
```

Your browser will launch the UI. Click **Boot System** to load the local Qwen model into VRAM.

---

## 📁 Repository Structure

```text
├── .gitignore                         # Prevents huge weights from bottlenecking Git
├── DATASETS/                          # Raw CSV supply chains for the 3 clients
├── sc50_logs/                         # Auto-generated JSONL/NDJSON decision logs from the LLM
├── app.py                             # Primary interactive Streamlit front-end
├── main.py                            # Backend configuration and federated logic
├── analyze_data.py                    # Legacy data validator
├── generate_datasets.py               # Generates numerical supply data
├── generate_finetune_dataset.py       # Converts CSV data into ChatML
└── finetune_qwen_supply_chain.py      # Unsloth Colab SFT training script
```

---

## 🤝 Acknowledgments

- Massive thanks to **Unsloth AI** for enabling fast 4-bit fine-tuning.
- Built using the **Qwen** foundation architectures.
- User interface powered by **Streamlit**.
