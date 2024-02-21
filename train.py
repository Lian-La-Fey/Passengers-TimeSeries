import torch

def train(model, criterion, optimizer, train_loader, val_loader, device, epochs=3000, max_patience=30):
    train_losses, val_losses = [], []

    # Early stopping variables
    best_val_loss = float('inf')
    best_model_state = None
    patience = 0

    for i in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for seq, labels in train_loader:
            seq, labels = seq.to(device), labels.to(device)
            optimizer.zero_grad()
            y_pred = model(seq)
            single_loss = criterion(y_pred, labels)
            single_loss.backward()
            optimizer.step()
            train_loss += single_loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for seq, labels in val_loader:
                seq, labels = seq.to(device), labels.to(device)
                y_pred = model(seq)
                val_loss += criterion(y_pred, labels).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        if i % 50 == 0 or i == 1:
            print(f'Epoch {i}/{epochs}, Training Loss: {avg_train_loss:.5f}, Validation Loss: {avg_val_loss:.5f}')

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            patience = 0
        else:
            patience += 1
            if patience >= max_patience:
                print(f'Epoch {i}/{epochs}, Training Loss: {avg_train_loss:.5f}, Validation Loss: {avg_val_loss:.5f}')
                print(f'Epoch {i - max_patience}/{epochs}, Training Loss: {train_losses[i - max_patience-1]:.5f}, Validation Loss: {val_losses[i - max_patience-1]:.5f}')
                print(f'Early stopping at epoch {i}, best model weights are saved at the {i - max_patience} epoch')
                break

    # Load the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return train_losses, val_losses
