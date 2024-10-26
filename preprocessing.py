from transformers import BertTokenizer
import torch

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('SI2M-Lab/DarijaBERT', do_lower_case=True)
def preprocessing_for_bert(data, tokenizer, max_len):
    input_ids = []
    attention_masks = []
    for sent in data:
        encoded_sent = tokenizer.encode_plus(
            text=sent,
            add_special_tokens=True,
            max_length=max_len,
            pad_to_max_length=True,
            return_attention_mask=True,
            truncation=True
        )
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))
    return torch.tensor(input_ids), torch.tensor(attention_masks)