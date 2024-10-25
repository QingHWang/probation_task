import argparse
import json
import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pandas as pd


def collate_fn_stage1(data):
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]
    data = tokenizer.batch_encode_plus(batch_text_or_text_pairs=sents,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=512,
                                   return_tensors='pt',
                                   return_length=True)
    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)
    token_type_ids = data['token_type_ids'].to(device)
    labels = torch.LongTensor(labels).to(device)
    return input_ids, attention_mask, token_type_ids, labels

def collate_fn_stage2(data):
    sents = [i[0] for i in data]
    labels = [i[1] for i in data]
    data = tokenizer.batch_encode_plus(batch_text_or_text_pairs=sents,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=125,
                                   return_tensors='pt',
                                   return_length=True)
    input_ids = data['input_ids'].to(device)
    attention_mask = data['attention_mask'].to(device)
    token_type_ids = data['token_type_ids'].to(device)
    labels = torch.LongTensor(labels).to(device)
    return input_ids, attention_mask, token_type_ids, labels

def prepare_data_stage1(path, seed):
    dataset = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            case = json.loads(line)
            fact = case['fact_d']
            label = case['label_1']
            dataset.append([fact, label])

        train_dataset, temp_dataset = train_test_split(dataset, test_size=0.4, random_state=seed)
        val_dataset, test_dataset = train_test_split(temp_dataset, test_size=0.5, random_state=seed)
        train_dataloader = DataLoader(dataset=train_dataset,
                    batch_size=16,
                    collate_fn=collate_fn_stage1,
                    shuffle=True,
                    drop_last=True)
        val_dataloader = DataLoader(dataset=val_dataset,
                    batch_size=16,
                    collate_fn=collate_fn_stage1,
                    shuffle=True,
                    drop_last=True)
        test_dataloader = DataLoader(dataset=test_dataset,
                    batch_size=32,
                    collate_fn = collate_fn_stage1,
                    shuffle=True,
                    drop_last=True)
        return train_dataloader, val_dataloader, test_dataloader, dataset


def prepare_data_stage2(predictions, path, seed):
    dataset = []
    with open(path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            case = json.loads(line)
            prob = case['huan_DES']
            label = case['label_2']
            if predictions[i] == 0:
                dataset.append([prob, label])

        train_dataset, temp_dataset = train_test_split(dataset, test_size=0.4, random_state=seed)
        val_dataset, test_dataset = train_test_split(temp_dataset, test_size=0.5, random_state=seed)
        train_dataloader = DataLoader(dataset=train_dataset,
                    batch_size=16,
                    collate_fn=collate_fn_stage2,
                    shuffle=True,
                    drop_last=True)
        val_dataloader = DataLoader(dataset=val_dataset,
                    batch_size=16,
                    collate_fn=collate_fn_stage2,
                    shuffle=True,
                    drop_last=True)
        test_dataloader = DataLoader(dataset=test_dataset,
                    batch_size=32,
                    collate_fn = collate_fn_stage2,
                    shuffle=True,
                    drop_last=True)
        return train_dataloader, val_dataloader, test_dataloader

def get_predictions(model, dataloader):
    model.eval()
    predictions = []
    for _, batch in enumerate(tqdm(dataloader, desc="Generating Predictions")):
        b_input_ids = batch[0]
        b_input_mask = batch[1]
        b_token_type_ids = batch[2]

        with torch.no_grad():
            outputs = model(b_input_ids, token_type_ids=b_token_type_ids, attention_mask=b_input_mask)
            logits = outputs[0]
            preds = np.argmax(logits.detach().cpu().numpy(), axis=1)
            predictions.extend(preds)

    return predictions


def train_model(model, train_dataloader, val_dataloader, epochs, learning_rate, save_path):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)
    best_val_accuracy = 0
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}"):
            b_input_ids, b_labels = batch
            model.zero_grad()
            outputs = model(b_input_ids, labels=b_labels)
            loss = outputs[0]
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss}")

        val_loss, val_accuracy = evaluate_model(model, val_dataloader)
        print(f"Validation Loss: {val_loss}")
        print(f"Validation Accuracy: {val_accuracy}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), save_path)

def evaluate_model(model, dataloader):
    model.eval()
    total_loss = 0
    predictions, true_labels = [], []
    
    for batch in dataloader:
        b_input_ids, b_labels = batch
        with torch.no_grad():
            outputs = model(b_input_ids, labels=b_labels)
            loss = outputs[0]
            logits = outputs[1]
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        predictions.extend(preds)
        true_labels.extend(b_labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    return avg_loss, accuracy

def test_model(model, dataloader):
    model.eval()
    predictions, true_labels = [], []
    
    for batch in dataloader:
        b_input_ids, b_labels = batch
        with torch.no_grad():
            outputs = model(b_input_ids)
            logits = outputs[0]
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        predictions.extend(preds)
        true_labels.extend(b_labels.cpu().numpy())
    
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='macro')
    recall = recall_score(true_labels, predictions, average='macro')
    f1 = f1_score(true_labels, predictions, average='macro')
    return accuracy, precision, recall, f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TwoStep-task learning with BERT, ELECTRA, or ALBERT")
    parser.add_argument('--model_type', type=str, default="BERT", choices=["BERT", "ELECTRA", "ALBERT"],
                        help="Choose between 'BERT', 'ELECTRA', or 'ALBERT'")
    parser.add_argument('--data_path', type=str, default="/root/works/WQH/prob_task/data/data_processed_vector.txt", help="Path to dataset")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs")
    parser.add_argument('--learning_rate', type=float, default=2e-5, help="Learning rate")
    parser.add_argument('--seeds', type=int, nargs='+', default=[42,52,62,72,82,92], help="List of random seeds for different runs")

    args = parser.parse_args()
    model_type = args.model_type
    data_path = args.data_path
    epochs = args.epochs
    learning_rate = args.learning_rate
    seeds = args.seeds

    model_path = {
        "BERT": "/root/works/WQH/prob_task/Model_base/bert-base-chinese/",
        "ELECTRA": "/root/works/WQH/prob_task/Model_base/electra-base-chinese/",
        "ALBERT": "/root/works/WQH/prob_task/Model_base/albert-base-chinese/"
    }.get(model_type, "/root/works/WQH/prob_task/Model_base/bert-base-chinese/")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    results_stage1 = []
    results_stage2 = []

    for i, seed in enumerate(seeds):
        print('-'*86)
        print(f'Round {i + 1} / {len(seeds)} Stage 1: Training on label_1...')

        train_dataloader, val_dataloader, test_dataloader = prepare_data_stage1(data_path, seed)
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path)
        model.to(device)

        train_model(model, train_dataloader, val_dataloader, epochs, learning_rate, f"{model_path}_stage1_{seed}.pt")

        # Testing Stage 1
        accuracy, precision, recall, f1 = test_model(model, test_dataloader)
        results_stage1.append({
            "Seed": seed,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

        # Get predictions for Stage 2
        predictions = get_predictions(model, test_dataloader)

        # Stage 2: Train on prob for instances where label_1 was predicted as 1
        print('-'*86)
        print(f'Round {len(results_stage2) + 1} / {len(seeds)} Stage 2: Training on label_2...')
        train_dataloader, val_dataloader, test_dataloader = prepare_data_stage2(predictions, data_path, seed)

        model_stage2 = BertForSequenceClassification.from_pretrained(model_path)
        model_stage2.to(device)
        train_model(model_stage2, train_dataloader, val_dataloader, epochs, learning_rate, f"{model_path}_stage2_{seed}.pt")

        # Testing Stage 2
        accuracy, precision, recall, f1 = test_model(model_stage2, test_dataloader)
        results_stage2.append({
            "Seed": seed,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

    # 输出表格
    df_stage1 = pd.DataFrame(results_stage1)
    df_stage2 = pd.DataFrame(results_stage2)
    
    print('-'*86)
    print("Stage 1 Results:")
    print(df_stage1)
    print('-'*86)
    print("Stage 2 Results:")
    print(df_stage2)

