import argparse
import pandas as pd
import torch
from Model_our import load_model_and_tokenizer
from pre_dataset import pre_dataset
from train_model import train_model
from test_model import test_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-task-CAT learning with BERT, ELECTRA, or ALBERT")
    parser.add_argument('--model_type', type=str, default="ALBERT", choices=["BERT", "ELECTRA", "ALBERT"],
                        help="Choose between 'BERT', 'ELECTRA', or 'ALBERT'")
    parser.add_argument('--task_type', type=str, default="MTC", choices=["MTC", "MTV", "STC", "STV"],
                        help="Choose between 'MTC', 'MTV', 'STC' or 'STV'")
    parser.add_argument('--data_path', type=str, default="/root/works/WQH/prob_task/data/data_processed_vector.txt", help="Path to dataset")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs")
    parser.add_argument('--learning_rate', type=float, default=2e-5, help="Learning rate")
    parser.add_argument('--seeds', type=list, nargs='+', default=[42], help="List of random seeds for different runs")
    parser.add_argument('--GPU_ID', type=int, default=0, help="GPU to use")
    args = parser.parse_args()
    seeds = args.seeds
    repeat_round = len(seeds)
    epochs = args.epochs
    learning_rate = args.learning_rate
    data_path = args.data_path
    device = torch.device(f'cuda:{args.GPU_ID}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    # model, tokenizer = load_model_and_tokenizer(args.task_type, args.model_type)
    # model.to(device)

    RESULT_test = []
    for i,seed in enumerate(seeds):
        model, tokenizer = load_model_and_tokenizer(args.task_type, args.model_type)
        model.to(device)
        model_best_path = f"/root/works/WQH/prob_task/Model_best/{args.model_type}-{args.task_type}-{i}.pt"
        train_dataloader, val_dataloader, test_dataloader = pre_dataset(data_path, seed, tokenizer, device)

        print('-'*86)
        print(f'Round {i + 1} / {repeat_round} Training...')
        train_model(model, train_dataloader, val_dataloader, epochs, learning_rate, device, model_best_path, args.task_type)
    
        print('-'*86)
        print('Testing Model')

        model.load_state_dict(torch.load(model_best_path, map_location=device))
        model.to(device)

        result_test = test_model(model, test_dataloader, args.task_type)
        RESULT_test.append(result_test)
    print('-'*86)
    with open(f"/root/works/WQH/prob_task/result_metric/{args.model_type}-{args.task_type}.txt", "w") as file:
        for i, data in enumerate(RESULT_test):
            if args.task_type == 'MTC' or args.task_type == 'MTV':
                data_dict = {
                    "Metric": ["Accuracy", "Mean Precision", "Mean Recall", "F1 Score", "Loss"],
                    "Main Task": [data["acc_main"], data["mp_main"], data["mr_main"], data["f1_main"], None],
                    "Auxiliary Task": [data["acc_aux"], data["mp_aux"], data["mr_aux"], data["f1_aux"], data["loss"]],
                }
            else:
                data_dict = {
                    "Metric": ["Accuracy", "Mean Precision", "Mean Recall", "F1 Score", "Loss"],
                    "Main Task": [data["acc_main"], data["mp_main"], data["mr_main"], data["f1_main"], data["loss"]],
                }
            df = pd.DataFrame(data_dict)
            file.write(f"Evaluation Metrics Table {i+1}\n")
            file.write(df.to_string(index=False))  
            file.write("\n\n")
            print(f"Evaluation Metrics Table {i+1}")
            print(df)
            print("\n")
    print("All tables have been saved in evaluation_metrics.txt")



