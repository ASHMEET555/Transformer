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

🏗️ Architecture Deep Dive
1️⃣ Transformer Model (model.py)

Multi-Head Attention (MHA)
Implements scaled dot-product attention across multiple heads to capture diverse linguistic relationships.

Positional Encoding
Fixed sinusoidal positional encodings to inject sequence order information.

Feed-Forward Network (FFN)
Position-wise fully connected layers with ReLU activation.

Residual Connections & Layer Normalization
Standard Add & Norm blocks for stable deep training.

Encoder–Decoder Stack

Encoder: 6 stacked layers

Decoder: 6 stacked layers

Cross-attention between source and target sequences

2️⃣ Data Pipeline (dataset.py)

BilingualDataset

Converts sentence pairs into tokenized tensors

Pads/truncates to fixed sequence length

Causal Masking

Prevents the decoder from attending to future tokens

Preserves autoregressive generation

Special Token Handling

[SOS], [EOS], [PAD] handled automatically

⚙️ Configuration (config.py)

Key hyperparameters (easily adjustable):

Parameter	Value
Batch Size	32
Sequence Length	128
d_model	512
Learning Rate	1e-4
Label Smoothing	0.1
Encoder Layers	6
Decoder Layers	6
🧪 Training & Evaluation
📦 Prerequisites

Python 3.10+

PyTorch

Hugging Face datasets & tokenizers

torchmetrics

tqdm

TensorBoard

Install dependencies:

pip install torch datasets tokenizers torchmetrics tqdm tensorboard
▶️ Training

Start training from scratch or resume from the latest checkpoint:

python train.py

The script automatically detects:

✅ CUDA (NVIDIA GPUs)

✅ MPS (Apple Silicon)

✅ CPU fallback

📊 Monitoring & Metrics

TensorBoard Integration

Training & validation loss

Evaluation metrics per epoch

Automated Validation Metrics

BLEU Score

Word Error Rate (WER)

Character Error Rate (CER)

👁️ Visualization & Inference

attention_visual.ipynb

Visualize attention maps across heads and layers

inference.ipynb

Interactive notebook for real-time translation testing

💾 Checkpointing

Saves:

Model weights

Optimizer state

Training epoch

Enables seamless training resume

📚 Dataset

OPUS Books Dataset

Clean, parallel English–Italian sentence pairs

Ideal for sentence-level translation tasks

🧠 Learning Objectives

This project is ideal if you want to:

Understand Transformers at a mathematical & implementation level

Build NMT systems without high-level abstractions

Explore attention mechanisms visually

Strengthen PyTorch and NLP fundamentals

📌 Future Improvements

Beam search decoding

Byte-Pair Encoding (BPE)

Transformer variants (Pre-LN, RoPE, etc.)

Mixed precision training

Multi-GPU training support

📄 References

Vaswani et al., Attention Is All You Need, 2017

OPUS: Open Parallel Corpus

PyTorch Documentation

⭐ Acknowledgements

Inspired by the original Transformer paper and modern NLP research.
Built with a focus on clarity, correctness, and learning.
