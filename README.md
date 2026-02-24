# Transformer from Scratch: Bilingual Language Translator 🇬🇧➡️🇮🇹

A **ground-up implementation of the Transformer architecture** as proposed in  
**_“Attention Is All You Need” (Vaswani et al.)**, built entirely from scratch using **PyTorch**.

This project performs **Neural Machine Translation (NMT)** for **English → Italian** using the **OPUS Books dataset**, with a strong emphasis on mathematical correctness, modular design, and interpretability.

---

## 🚀 Highlights

- 🔧 Full Transformer implementation (no `nn.Transformer`)
- 🌍 English → Italian translation
- 🧠 Multi-Head Self-Attention & Cross-Attention
- 📐 Sinusoidal Positional Encoding
- 🔄 Encoder–Decoder architecture (6 layers each)
- 🧪 BLEU, WER, and CER evaluation
- 📊 TensorBoard monitoring
- 💾 Automatic checkpointing & resume support
- 👁️ Attention weight visualization

---

## 📂 Project Structure

```plaintext
.
├── config.py                 # Configuration parameters & paths
├── dataset.py                # Dataset processing & masking logic
├── model.py                  # Transformer architecture (Encoder, Decoder, MHA)
├── train.py                  # Training loop, validation & greedy decoding
├── tokenizer_en.json         # Pre-trained English WordLevel tokenizer
├── tokenizer_it.json         # Pre-trained Italian WordLevel tokenizer
├── attention_visual.ipynb    # Attention weight visualization
├── inference.ipynb           # Interactive translation testing
├── README.md                 # Project documentation
└── opus_books_weights/       # Model checkpoints (auto-generated)
```
---

## 🏗️ Architecture Deep Dive

### 1️⃣ Transformer Model (`model.py`)

#### 🔹 Multi-Head Attention (MHA)
Implements **scaled dot-product attention** across multiple heads to capture diverse syntactic and semantic relationships in language.

#### 🔹 Positional Encoding
Uses **fixed sinusoidal positional encodings** to inject word order information without introducing additional learned parameters.

#### 🔹 Feed-Forward Network (FFN)
Position-wise fully connected layers with **ReLU activation**, applied independently to each token representation.

#### 🔹 Residual Connections & Layer Normalization
Standard **Add & Norm** blocks ensure stable gradient flow and efficient deep training.

#### 🔹 Encoder–Decoder Stack
- **Encoder:** 6 stacked layers for source sentence encoding  
- **Decoder:** 6 stacked layers for autoregressive target generation  
- **Cross-Attention:** Enables the decoder to attend over encoded source representations  

---

### 2️⃣ Data Pipeline (`dataset.py`)

#### 🔹 BilingualDataset
- Converts raw English–Italian sentence pairs into tokenized tensors  
- Pads or truncates sequences to a fixed maximum length  

#### 🔹 Causal Masking
- Prevents the decoder from attending to future tokens  
- Preserves the autoregressive decoding property  

#### 🔹 Special Token Handling
Automatically manages:
- `[SOS]` — Start of sentence  
- `[EOS]` — End of sentence  
- `[PAD]` — Padding token  

---

## ⚙️ Configuration (`config.py`)

Key hyperparameters (fully configurable):

| Parameter         | Value |
|-------------------|-------|
| Batch Size        | 32    |
| Sequence Length   | 128   |
| d_model           | 512   |
| Learning Rate     | 1e-4  |
| Label Smoothing   | 0.1   |
| Encoder Layers    | 6     |
| Decoder Layers    | 6     |

---

## 🧪 Training & Evaluation

### 📦 Prerequisites
- Python **3.10+**
- PyTorch
- Hugging Face `datasets` & `tokenizers`
- `torchmetrics`
- `tqdm`
- TensorBoard

Install dependencies:


▶️ Training

Start training from scratch or resume from the latest checkpoint:

python train.py

The training script automatically detects and utilizes:

CUDA (NVIDIA GPUs)

MPS (Apple Silicon)

CPU fallback

📊 Monitoring & Metrics
🔹 TensorBoard Integration

Training loss

Validation loss

Evaluation metrics per epoch

🔹 Evaluation Metrics

BLEU Score

Word Error Rate (WER)

Character Error Rate (CER)

👁️ Visualization & Inference
🔹 attention_visual.ipynb

Visualize attention weights across layers and heads for interpretability.

🔹 inference.ipynb

Interactive notebook for real-time translation testing.

💾 Checkpointing

The training pipeline automatically saves:

Model weights

Optimizer state

Current training epoch

This enables seamless resumption of training.

📚 Dataset

OPUS Books Dataset

Clean, parallel English–Italian sentence pairs

Suitable for sentence-level neural machine translation

🧠 Learning Objectives

This project is designed to help you:

Understand Transformers at a mathematical and implementation level

Build NMT systems without relying on high-level abstractions

Explore attention mechanisms visually

Strengthen PyTorch and NLP fundamentals

📌 Future Improvements

Beam search decoding

Byte-Pair Encoding (BPE)

Transformer variants (Pre-LN, RoPE)

Mixed precision training

Multi-GPU / Distributed training

📄 References

Vaswani et al., Attention Is All You Need, 2017

OPUS: Open Parallel Corpus

PyTorch Documentation

⭐ Acknowledgements

Inspired by the original Transformer paper and modern NLP research.
Built with a focus on clarity, correctness, and learning.

If you find this project useful, consider starring ⭐ the repository!
