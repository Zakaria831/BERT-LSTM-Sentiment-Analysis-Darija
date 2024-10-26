# BERT-LSTM Sentiment Analysis for Darija

This project combines BERT embeddings and an LSTM layer to classify tweets written in Darija (Moroccan Arabic) as positive or negative. The model leverages the `SI2M-Lab/DarijaBERT` for high-quality Darija language embeddings, followed by an LSTM for sequential processing, making it effective for sentiment analysis on short, informal texts like tweets.

## Project Overview

- **Goal**: To classify Darija tweets as positive or negative using a hybrid BERT + LSTM model.
- **Model**: `SI2M-Lab/DarijaBERT` is used for extracting Darija-specific embeddings, followed by an LSTM layer and a custom attention mechanism.
- **Architecture**:
  - **BERT Embeddings**: The Darija-specific BERT model is used to produce token-level embeddings.
  - **LSTM Layer**: An LSTM processes these embeddings to capture sequential information in tweets.
  - **Attention Mechanism**: A custom attention layer highlights the most important features in each tweet.
- **Dataset**: Custom labeled dataset with Darija tweets.
- **Evaluation Metrics**: Accuracy, ROC-AUC, and a visualized ROC curve for model performance.

## Setup and Requirements

- Python 3.7+
- Libraries: `transformers`, `torch`, `pandas`, `matplotlib`, `seaborn`, `sklearn`, `wordcloud`

Install dependencies:

```bash
pip install -r requirements.txt
