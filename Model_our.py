
from transformers import BertModel, BertTokenizer, BertConfig
from transformers import ElectraModel, ElectraTokenizer, ElectraConfig
from transformers import AlbertModel, AlbertTokenizer, AlbertConfig
import torch.nn as nn
import torch

class MultiTaskCATModel(nn.Module):
    def __init__(self, model_type, config):
        super().__init__()

        if model_type == "BERT":
            self.model = BertModel(config)
        elif model_type == "ELECTRA":
            self.model = ElectraModel(config)
        elif model_type == "ALBERT":
            self.model = AlbertModel(config)
        else:
            raise ValueError("Unsupported model type. Choose either 'BERT', 'ELECTRA', or 'ALBERT'.")

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier_main = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.num_labels)
        )
        
        self.classifier_aux = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.num_labels)
        )
    
    def forward(self, input_ids_1, input_ids_2, 
                attention_mask_1=None, token_type_ids_1=None,
                attention_mask_2=None, token_type_ids_2=None,
                labels_main=None, labels_aux=None):

        outputs_1 = self.model(input_ids_1, attention_mask=attention_mask_1, token_type_ids=token_type_ids_1)
        pooled_output_1 = outputs_1.last_hidden_state[:, 0, :] 
        outputs_2 = self.model(input_ids_2, attention_mask=attention_mask_2, token_type_ids=token_type_ids_2)
        pooled_output_2 = outputs_2.last_hidden_state[:, 0, :]

        concatenated_embeddings = torch.cat((pooled_output_1, pooled_output_2), dim=1)
        pooled_output = self.dropout(concatenated_embeddings)
        logits_main = self.classifier_main(pooled_output)
        logits_aux = self.classifier_aux(pooled_output)

        loss = None
        if labels_main is not None and labels_aux is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss_main = loss_fct(logits_main.view(-1, 2), labels_main.view(-1))
            loss_aux = loss_fct(logits_aux.view(-1, 2), labels_aux.view(-1))
            loss = 0.7 * loss_main + 0.3 * loss_aux

        return (loss, logits_main, logits_aux)


class MultiTaskVectorModel(nn.Module):
    def __init__(self, model_type, config):
        super().__init__()

        if model_type == "BERT":
            self.model = BertModel(config)
        elif model_type == "ELECTRA":
            self.model = ElectraModel(config)
        elif model_type == "ALBERT":
            self.model = AlbertModel(config)
        else:
            raise ValueError("Unsupported model type. Choose either 'BERT', 'ELECTRA', or 'ALBERT'.")

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier_main = nn.Linear(config.hidden_size + 44, 2)  
        self.classifier_aux = nn.Linear(config.hidden_size + 44, 2)  

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, 
                labels_main=None, labels_aux=None, add_input = None):
        outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        concatenated_embeddings = torch.cat((pooled_output, add_input), dim=1)
        pooled_output = self.dropout(concatenated_embeddings)

        logits_main = self.classifier_main(pooled_output)
        logits_aux = self.classifier_aux(pooled_output)

        loss = None
        if labels_main is not None and labels_aux is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss_main = loss_fct(logits_main.view(-1, 2), labels_main.view(-1))
            loss_aux = loss_fct(logits_aux.view(-1, 2), labels_aux.view(-1))
            loss = 0.7*loss_main + 0.3*loss_aux

        return (loss, logits_main, logits_aux)
    


class SingleTaskCATModel(nn.Module):
    def __init__(self, model_type, config):
        super().__init__()

        if model_type == "BERT":
            self.model = BertModel(config)
        elif model_type == "ELECTRA":
            self.model = ElectraModel(config)
        elif model_type == "ALBERT":
            self.model = AlbertModel(config)
        else:
            raise ValueError("Unsupported model type. Choose either 'BERT', 'ELECTRA', or 'ALBERT'.")

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.num_labels)
        )  
        
    
    def forward(self, input_ids_1, input_ids_2, 
                attention_mask_1=None, token_type_ids_1=None,
                attention_mask_2=None, token_type_ids_2=None,
                labels=None):

        outputs_1 = self.model(input_ids_1, attention_mask=attention_mask_1, token_type_ids=token_type_ids_1)
        pooled_output_1 = outputs_1.last_hidden_state[:, 0, :] 
        outputs_2 = self.model(input_ids_2, attention_mask=attention_mask_2, token_type_ids=token_type_ids_2)
        pooled_output_2 = outputs_2.last_hidden_state[:, 0, :]

        concatenated_embeddings = torch.cat((pooled_output_1, pooled_output_2), dim=1)
        pooled_output = self.dropout(concatenated_embeddings)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return (loss, logits)


class SingleTaskVectorModel(nn.Module):
    def __init__(self, model_type, config):
        super().__init__()

        if model_type == "BERT":
            self.model = BertModel(config)
        elif model_type == "ELECTRA":
            self.model = ElectraModel(config)
        elif model_type == "ALBERT":
            self.model = AlbertModel(config)
        else:
            raise ValueError("Unsupported model type. Choose either 'BERT', 'ELECTRA', or 'ALBERT'.")

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size + 44, 2)   

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, 
                labels=None, add_input = None):
        outputs = self.model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        concatenated_embeddings = torch.cat((pooled_output, add_input), dim=1)
        pooled_output = self.dropout(concatenated_embeddings)

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            loss_fn = torch.nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return (loss, logits)
    


def load_model_and_tokenizer(task_type, model_type):
    if model_type == "BERT":
        model_path = "/root/works/WQH/prob_task/Model_base/bert-base-chinese"
        tokenizer = BertTokenizer.from_pretrained(model_path)
        config = BertConfig.from_pretrained(model_path)
        if task_type == "MTC":
            model = MultiTaskCATModel(model_type, config)
        elif task_type == "MTV":
            model = MultiTaskVectorModel(model_type, config)
        elif task_type == "STC":
            model = SingleTaskCATModel(model_type, config)
        elif task_type == "STV":
            model = SingleTaskVectorModel(model_type, config)
        else:
            raise ValueError("Unsupported Task type. Choose either 'MTC', 'MTV', 'STC', or 'STV'.")

    elif model_type == "ELECTRA":
        model_path = "/root/works/WQH/prob_task/Model_base/electra-base-chinese"
        tokenizer = ElectraTokenizer.from_pretrained(model_path)
        config = ElectraConfig.from_pretrained(model_path)
        if task_type == "MTC":
            model = MultiTaskCATModel(model_type, config)
        elif task_type == "MTV":
            model = MultiTaskVectorModel(model_type, config)
        elif task_type == "STC":
            model = SingleTaskCATModel(model_type, config)
        elif task_type == "STV":
            model = SingleTaskVectorModel(model_type, config)
        else:
            raise ValueError("Unsupported Task type. Choose either 'MTC', 'MTV', 'STC', or 'STV'.")

    elif model_type == "ALBERT":
        model_path = "/root/works/WQH/prob_task/Model_base/albert-base-chinese"
        tokenizer = BertTokenizer.from_pretrained(model_path)
        config = AlbertConfig.from_pretrained(model_path)
        if task_type == "MTC":
            model = MultiTaskCATModel(model_type, config)
        elif task_type == "MTV":
            model = MultiTaskVectorModel(model_type, config)
        elif task_type == "STC":
            model = SingleTaskCATModel(model_type, config)
        elif task_type == "STV":
            model = SingleTaskVectorModel(model_type, config)
        else:
            raise ValueError("Unsupported Task type. Choose either 'MTC', 'MTV', 'STC', or 'STV'.")

    else:
        raise ValueError("Unsupported model type. Choose either 'BERT', 'ELECTRA', or 'ALBERT'.")
    
    return model, tokenizer






