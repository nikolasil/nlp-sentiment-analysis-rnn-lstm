# Sequential Sentiment Analysis: From Vanilla RNN to LSTM/GRU

This project explores the evolution of Recurrent Neural Networks (RNN) in the context of NLP. I conducted a multi-stage experiment to address the **Vanishing Gradient Problem** and improve the model's "memory" when processing tweet sequences.

## 🧠 Experimental Framework

All models utilize **GloVe 300d embeddings** and a standardized preprocessing pipeline:
* **Cleaning:** URL/Punctuation removal & Emoticon filtering.
* **Normalization:** Tokenization followed by Lemmatization to reduce sparsity.
* **Architecture:** Comparative analysis of **Vanilla RNN**, **GRU**, and **LSTM** cells.

---

## 📈 Stage 1: The Vanilla RNN Baseline
The initial experiment focused on scaling the hidden size of a standard RNN.

### Finding the Information Bottleneck
While increasing the `hidden_size` from **32 to 256** improved the F1 score, further scaling to **1024** only served to increase overfitting without performance gains.

<p align="center">
  <img src="images/1.2.1.png" width="230" title="Hidden 256" />
  <img src="images/1.3.1.png" width="230" title="Hidden 1024 (Overfit)" />
</p>

**Technical Insight:** Vanilla RNNs suffered from **Short-Term Memory**. Even with stacked layers, the model could not effectively propagate gradients through longer sequences.

---

## ⚡ Stage 2: Addressing Memory with GRU & LSTM
To solve the memory bottleneck, I implemented gated architectures (Gated Recurrent Units and Long Short-Term Memory).



### GRU Experiments
Implemented **Bidirectional GRUs** to capture context from both ends of a tweet. While the loss stabilized, the model was highly sensitive to the number of layers.

<p align="center">
  <img src="images/2.2.1.png" width="230" title="Bidirectional GRU" />
</p>

### LSTM: The Optimal Performer
Transitioning to **LSTM cells** with **AMSGrad** optimization provided the most robust results. LSTMs effectively mitigated the "short-term memory" issue, resulting in the highest F1 score of the entire study.

<p align="center">
  <img src="images/2.5.1.png" width="230" title="Final LSTM Loss" />
  <img src="images/2.5.2.png" width="230" title="Final LSTM F1" />
  <img src="images/2.5.3.png" width="230" title="Final LSTM ROC" />
</p>

---

## 🏆 Final Conclusion & Comparisons

Through exhaustive testing, I compared these sequential models against previous FeedForward (FFNN) attempts.

| Model Type | Best Loss | Best F1 Score | Key Trait |
| :--- | :--- | :--- | :--- |
| **FFNN (Ex 2)** | 0.15 | 0.61 | Fast, but lacks context |
| **LSTM (Ex 3)** | 0.75 | **0.65** | Slower, but captures sequence |

**Key Takeaway:** Despite a higher numerical loss, the **LSTM model achieved superior classification accuracy (F1)**. This demonstrates that for NLP, capturing the *order* and *context* of words is more critical than simply minimizing the error of independent word counts.

---
## 🎓 Academic Context
Developed as part of the Artificial Intelligence II course at the National and Kapodistrian University of Athens (UoA). Based on the UC Berkeley CS188 framework.

## 🚦 Setup & Usage
```bash
# Install dependencies
pip install torch pandas scikit-learn nltk numpy matplotlib

# Run the sequential trainer
python main.py

