import random
from sklearn.model_selection import train_test_split
import torch
import json
from torch.utils.data import DataLoader

def collate_fn(data,token, device):
    sents_1 = [i[0] for i in data]
    sents_2 = [i[1] for i in data]
    Vectors = [i[2] for i in data]
    labels_main = [i[3] for i in data]
    labels_aux = [i[4] for i in data]

    data_1 = token.batch_encode_plus(batch_text_or_text_pairs=sents_1,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=512,
                                   return_tensors='pt',
                                   return_length=True)

    data_2 = token.batch_encode_plus(batch_text_or_text_pairs=sents_2,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=125,
                                   return_tensors='pt',
                                   return_length=True)

    input_ids_1 = data_1['input_ids'].to(device)
    input_ids_2 = data_2['input_ids'].to(device)
    attention_mask_1 = data_1['attention_mask'].to(device)
    attention_mask_2 = data_2['attention_mask'].to(device)
    token_type_ids_1 = data_1['token_type_ids'].to(device)
    token_type_ids_2 = data_2['token_type_ids'].to(device)
    Vectors_input = torch.tensor(Vectors).float().to(device)
    labels_main = torch.LongTensor(labels_main).to(device)
    labels_aux = torch.LongTensor(labels_aux).to(device)
    return (input_ids_1, attention_mask_1, token_type_ids_1), (input_ids_2, attention_mask_2, token_type_ids_2), Vectors_input, labels_main, labels_aux

def pre_dataset(data_path, seed, token, device):
    dataset = []
    all_dataset = []   
    with open(data_path, 'r', encoding='utf-8') as file:
        for line in file:
            case = json.loads(line)
            fact = case['fact_d']
            description = case['huan_DES']
            vectors = case['factor_V']
            label_main = case['label_2']
            label_aux = case['label_1']
            all_dataset.append([fact, description, vectors, label_main, label_aux])

    label_main_0 = [data for data in all_dataset if data[3] == 0]
    label_main_1 = [data for data in all_dataset if data[3] == 1]
    # print(len(label_main_0),len(label_main_1))
    sample_size = min(len(label_main_0), len(label_main_1))
    sampled_label_main_0 = random.sample(label_main_0, sample_size)
    sampled_label_main_1 = random.sample(label_main_1, sample_size)
    dataset = sampled_label_main_0 + sampled_label_main_1
    train_dataset, temp_dataset = train_test_split(dataset, test_size=0.4, random_state=seed)
    val_dataset, test_dataset = train_test_split(temp_dataset, test_size=0.5, random_state=seed)
    train_dataloader = DataLoader(train_dataset, batch_size=16, collate_fn=lambda batch: collate_fn(batch, token, device), shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, collate_fn=lambda batch: collate_fn(batch, token, device), shuffle=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, collate_fn=lambda batch: collate_fn(batch, token, device), shuffle=True, drop_last=True)
    return train_dataloader, val_dataloader, test_dataloader


