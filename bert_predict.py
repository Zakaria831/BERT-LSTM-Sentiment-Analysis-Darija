import torch.nn.functional as F
import torch


# Define the bert_predict function
def bert_predict(model, test_dataloader):
    """Perform a forward pass on the trained BERT model to predict probabilities on the test set."""
    model.eval()
    all_logits = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for batch in test_dataloader:
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]

        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        all_logits.append(logits)

    # Concatenate all logits and apply softmax to get probabilities
    all_logits = torch.cat(all_logits, dim=0)
    probs = torch.nn.functional.softmax(all_logits, dim=1).cpu().numpy()

    return probs