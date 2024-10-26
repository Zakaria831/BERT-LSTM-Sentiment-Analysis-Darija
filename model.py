import torch.nn as nn
from transformers import BertModel
import numpy as np
import torch

class BertLSTMClassifier(nn.Module):
    def __init__(self, freeze_bert=False):
        super(BertLSTMClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('SI2M-Lab/DarijaBERT')
        self.lstm = nn.LSTM(768, 128, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(256, 1)
        self.layer_norm = nn.LayerNorm(256)
        self.classifier = nn.Sequential(
            nn.Linear(256, 50),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(50, 2)
        )

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        lstm_output, _ = self.lstm(last_hidden_state)
        attn_weights = torch.softmax(self.attention(lstm_output), dim=1)
        context_vector = torch.sum(attn_weights * lstm_output, dim=1)
        context_vector = self.layer_norm(context_vector)
        logits = self.classifier(context_vector)
        return logits