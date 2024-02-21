import torch

def test(model, criterion, test_loader, device):
    test_loss = 0.0
    with torch.inference_mode():
        for seq, labels in test_loader:
            seq, labels = seq.to(device), labels.to(device)
            y_pred = model(seq)
            test_loss += criterion(y_pred, labels).item()

    avg_test_loss = test_loss / len(test_loader)
    print(f'Test Loss: {avg_test_loss:.8f}')