import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
import torch

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def test_model(model, test_dataloader, task_type):

    model.eval()
    total_test_accuracy_main = 0
    total_test_accuracy_aux = 0
    total_test_loss = 0

    all_logits_main = []
    all_logits_aux = []
    all_label_ids_main = []
    all_label_ids_aux = []

    for batch_1, batch_2, b_Vectors_input, b_labels_main, b_labels_aux in test_dataloader:
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
        total_test_loss += loss.item()  
        logits_main = logits_main.detach().cpu().numpy()
        label_ids_main = b_labels_main.to('cpu').numpy()
        all_logits_main.extend(logits_main)
        all_label_ids_main.extend(label_ids_main)
        total_test_accuracy_main += flat_accuracy(logits_main, label_ids_main)
        avg_test_accuracy_main = total_test_accuracy_main / len(test_dataloader)

        if task_type == 'MTC' or task_type == 'MTV':
            logits_aux = logits_aux.detach().cpu().numpy()
            label_ids_aux = b_labels_aux.to('cpu').numpy()
            all_logits_aux.extend(logits_aux)
            all_label_ids_aux.extend(label_ids_aux)
            total_test_accuracy_aux += flat_accuracy(logits_aux, label_ids_aux)
            avg_test_accuracy_aux = total_test_accuracy_aux / len(test_dataloader)

    
    avg_test_loss = total_test_loss / len(test_dataloader)
    all_predictions_main = np.argmax(all_logits_main, axis=1)
    if task_type == 'MTC' or task_type == 'MTV':
        all_predictions_aux = np.argmax(all_logits_aux, axis=1)

    precision_main = precision_score(all_label_ids_main, all_predictions_main, average='macro')
    recall_main = recall_score(all_label_ids_main, all_predictions_main, average='macro')
    f1_main = f1_score(all_label_ids_main, all_predictions_main, average='macro')

    if task_type == 'MTC' or task_type == 'MTV':
        precision_aux = precision_score(all_label_ids_aux, all_predictions_aux, average='macro')
        recall_aux = recall_score(all_label_ids_aux, all_predictions_aux, average='macro')
        f1_aux = f1_score(all_label_ids_aux, all_predictions_aux, average='macro')

    if task_type == 'MTC' or task_type == 'MTV':
        return {"acc_main": avg_test_accuracy_main, 
                "mp_main": precision_main, 
                "mr_main": recall_main, 
                "f1_main": f1_main, 
                "acc_aux": avg_test_accuracy_aux, 
                "mp_aux": precision_aux, 
                "mr_aux": recall_aux, 
                "f1_aux": f1_aux, 
                "loss": avg_test_loss
                }
    else:
        return {"acc_main": avg_test_accuracy_main, 
                "mp_main": precision_main, 
                "mr_main": recall_main, 
                "f1_main": f1_main,  
                "loss": avg_test_loss
                }