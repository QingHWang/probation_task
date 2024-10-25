import numpy as np
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def eval_model(model, val_dataloader, task_type):
    model.eval()
    total_eval_accuracy_main = 0
    total_eval_loss = 0

    for batch_1, batch_2, b_Vectors_input, b_labels_main, b_labels_aux in val_dataloader:
        b_input_ids_1, b_input_mask_1, b_token_type_ids_1 = batch_1
        b_input_ids_2, b_input_mask_2, b_token_type_ids_2 = batch_2

        with torch.no_grad():
            if task_type == 'MTC':
                loss, logits_main, logits_aux = model(b_input_ids_1, b_input_ids_2,
                                                  b_input_mask_1, b_token_type_ids_1,
                                                  b_input_mask_2, b_token_type_ids_2,
                                                  b_labels_main, b_labels_aux)
            elif task_type == 'MTV':
                loss, logits_main, logits_aux = model(b_input_ids_1, b_input_mask_1, b_token_type_ids_1, 
                                                  b_labels_main, b_labels_aux, b_Vectors_input)
            elif task_type == 'STC':
                loss, logits_main = model(b_input_ids_1, b_input_ids_2,
                                        b_input_mask_1, b_token_type_ids_1,
                                        b_input_mask_2, b_token_type_ids_2,
                                        b_labels_main)
            elif task_type == 'STV':
                loss, logits_main = model(b_input_ids_1, b_input_mask_1, b_token_type_ids_1, 
                                        b_labels_main, b_Vectors_input)
                
            total_eval_loss += loss.item()

            logits_main = logits_main.detach().cpu().numpy()
            label_ids_main = b_labels_main.to('cpu').numpy()
            total_eval_accuracy_main += flat_accuracy(logits_main, label_ids_main)

        avg_val_accuracy_main = total_eval_accuracy_main / len(val_dataloader)
        avg_val_loss = total_eval_loss / len(val_dataloader)
    return avg_val_loss, avg_val_accuracy_main


def train_model(model, train_dataloader, validation_dataloader, epochs, learning_rate, device, best_model_path, task_type):
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    model.to(device)
    best_accuracy = 0
    for epoch in range(epochs):
        total_train_accuracy_main = 0
        total_train_loss = 0
        model.train()

        for batch_1, batch_2, b_Vectors_input, b_labels_main, b_labels_aux in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} is Training"):
            b_input_ids_1, b_input_mask_1, b_token_type_ids_1 = batch_1
            b_input_ids_2, b_input_mask_2, b_token_type_ids_2 = batch_2

            model.zero_grad()
            if task_type == 'MTC':
                loss, logits_main, logits_aux = model(b_input_ids_1, b_input_ids_2,
                                                  b_input_mask_1, b_token_type_ids_1,
                                                  b_input_mask_2, b_token_type_ids_2,
                                                  b_labels_main, b_labels_aux)
            elif task_type == 'MTV':
                loss, logits_main, logits_aux = model(b_input_ids_1, b_input_mask_1, b_token_type_ids_1, 
                                                  b_labels_main, b_labels_aux, b_Vectors_input)
            elif task_type == 'STC':
                loss, logits_main = model(b_input_ids_1, b_input_ids_2,
                                        b_input_mask_1, b_token_type_ids_1,
                                        b_input_mask_2, b_token_type_ids_2,
                                        b_labels_main)
            elif task_type == 'STV':
                loss, logits_main = model(b_input_ids_1, b_input_mask_1, b_token_type_ids_1, 
                                        b_labels_main, b_Vectors_input)
            total_train_loss += loss.item()
            loss.backward()
            logits_main = logits_main.detach().cpu().numpy()
            label_ids_main = b_labels_main.to('cpu').numpy()
            total_train_accuracy_main += flat_accuracy(logits_main, label_ids_main)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        avg_train_accuracy_main = total_train_accuracy_main / len(train_dataloader)
        val_loss, val_accuracy = eval_model(model, validation_dataloader, task_type)
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)
        
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.2f}, Training Acc: {avg_train_accuracy_main*100:.2f}%, Validation Loss: {val_loss:.2f}, Validation Accuracy: {val_accuracy*100:.2f}%")

    print("Training complete.")