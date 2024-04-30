import numpy as np
import torch
from sklearn.metrics import f1_score
from .utils import adjacency_matrix, calculate_mat, MADGap   

def training_pipeline_cora(
        model, 
        loss_fcn, 
        optimizer, 
        data, 
        device, 
        lambda_reg=0,  # Regularization coefficient
        max_epochs=100,
        verbose=True):

    if verbose : 
        print("=====================================")
        print(f"Training the {model.__class__.__name__} model for {len(model.convs)} layers with MADReg = {lambda_reg}")
        print("=====================================")
        print("Starting training...")

    best_val_score = float('-inf')
    epochs_without_improvement = 0
    early_stopping_patience = 10
    mad_gaps = []

    data = data[0].to(device)

    for epoch in range(max_epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = loss_fcn(out[data.train_mask], data.y[data.train_mask])

        if lambda_reg > 0:
            adj_matrix = adjacency_matrix(data.edge_index, data.num_nodes).to(device)
            embeddings = model(data.x, data.edge_index)
            mat_neib, mat_far = calculate_mat(adj_matrix, 3, 8)
            _, _, mad_gap = MADGap(embeddings, mat_neib, mat_far)
            loss -= lambda_reg * mad_gap  # Adding MADReg penalization

        loss.backward()
        optimizer.step()
        
        # Evaluate
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
            correct = pred[data.val_mask] == data.y[data.val_mask]
            val_acc = int(correct.sum()) / int(data.val_mask.sum())
            val_loss = loss_fcn(out[data.val_mask], data.y[data.val_mask]).item()
        
        if epoch % 50 == 0 and verbose:
            if lambda_reg > 0:
                print(f"Epoch {epoch + 1:05d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.4f} | MADGap: {mad_gap:.4f}")
            else:
                print(f"Epoch {epoch + 1:05d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.4f}")

        if val_acc > best_val_score:
            best_val_score = val_acc
            epochs_without_improvement = 0  # Reset counter
        else:
            epochs_without_improvement += 1  # Increment counter

        # Check for early stopping condition
        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs due to no improvement in validation accuracy.")
            break

    return model


def test_cora(model, data, device):
    data = data.to(device)
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        correct = pred[data.test_mask] == data.y[data.test_mask]
        test_acc = int(correct.sum()) / int(data.test_mask.sum())
        print(f"Test Accuracy: {test_acc:.4f}")
    return test_acc




