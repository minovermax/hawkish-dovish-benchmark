from transformers import BertForSequenceClassification, BertTokenizer, RobertaTokenizerFast, RobertaForSequenceClassification, BertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification, XLNetForSequenceClassification, XLNetTokenizerFast
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import pandas as pd
import os


def fine_tune_plm(gpu_numbers: str, seed: int, feature: str, save_model_path: str):
    """
    Fine-tunes a pre-trained language model (PLM) for sequence classification on a specific feature.
    Args:
        gpu_numbers (str): Comma-separated string of GPU numbers to use.
        seed (int): Random seed for reproducibility.
        feature (str): The target feature/label column in the dataset (e.g., 'hawkish', 'forward_looking').
        save_model_path (str): Path to save the fine-tuned model and tokenizer.
    Returns:
        list: Experiment results including training and testing metrics.
    """
    # GPU setup
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_numbers
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # TODO: Load the dataset
    # Example: Load `data.csv` containing sentences, sentiment_label, time_label, certain_label
    # data = pd.read_csv("data.csv")

    # TODO: Split data using train_test_split
    # Example:
    # train_data, temp_data = train_test_split(data, test_size=0.2, random_state=seed)
    # val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=seed)

    # TODO: Select a pre-trained model and tokenizer
    model = None
    tokenizer = None
    
    # Preprocessing
    def preprocess_data(dataset, feature):
        """
        Preprocesses the dataset for tokenization.
        Args:
            dataset (pd.DataFrame): The dataset containing the sentences and labels.
            feature (str): The target column to use as labels.
        Returns:
            TensorDataset: Dataset ready for DataLoader.
        """
        # TODO: Extract the sentences and labels from the dataset for the specific feature
        sentences = None
        labels = None
        
        # Tokenizing the sentences and return a TensorDataset
        tokens = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True, max_length=256)
        return TensorDataset(tokens['input_ids'], tokens['attention_mask'], torch.LongTensor(labels))

    # TODO: Preprocess the train, test, and val datasets using the preprocess_data function

    # TODO: Create DataLoaders
    # Example:
    # train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True) # Change batch_size if needed
    # val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # TODO: Define optimizer
    # Example. Change the learning rate if needed
    # optimizer = optim.AdamW(model.parameters(), lr=1e-5)

    # Training and Validation Loop
    max_num_epochs = 20
    early_stopping_count = 0
    best_loss = float('inf')

    for epoch in range(max_num_epochs):
        # TODO: Implement training logic
        # Example:
        # model.train()
        # for batch in train_dataloader:
        #     batch = [b.to(device) for b in batch]
        #     inputs, masks, labels = batch
        #     optimizer.zero_grad()
        #     outputs = model(inputs, attention_mask=masks, labels=labels)
        #     loss = outputs.loss
        #     loss.backward()
        #     optimizer.step()

        # TODO: Implement validation logic
        # Example:
        # model.eval()
        # val_loss, val_f1, val_precision, val_recall = 0, 0, 0, 0
        # for batch in val_dataloader:
        #     batch = [b.to(device) for b in batch]
        #     inputs, masks, labels = batch
        #     outputs = model(inputs, attention_mask=masks, labels=labels)
        #     val_loss += outputs.loss.item()
        #     TODO: Compute F1, precision, and recall for validation
        #     preds = torch.argmax(outputs.logits, dim=1)
        #     val_f1 += f1_score(labels.cpu(), preds.cpu(), average='weighted')
        #     val_precision += precision_score(labels.cpu(), preds.cpu(), average='weighted')
        #     val_recall += recall_score(labels.cpu(), preds.cpu(), average='weighted')
        
        # val_loss /= len(val_dataloader)  # Average validation loss
        # val_f1 /= len(val_dataloader)   # Average F1 score
        # val_precision /= len(val_dataloader)  # Average precision
        # val_recall /= len(val_dataloader)  # Average recall
        # print(f"Validation Loss: {val_loss}, F1: {val_f1}, Precision: {val_precision}, Recall: {val_recall}")

        # TODO: Update early stopping counter
        # if val_loss < best_loss:
        #     best_loss = val_loss
        #     early_stopping_count = 0
        #     torch.save({'model_state_dict': model.state_dict()}, 'best_model.pt')
        # else:
        #     early_stopping_count += 1

        # TODO:
        # Add logging/print statements to monitor training progress
        
        # Break if early stopping condition is met
        if early_stopping_count >= 5:
            break

    # TODO: Load the best model
    # Example:
    # checkpoint = torch.load('best_model.pt')
    # model.load_state_dict(checkpoint['model_state_dict'])

    # Testing
    # Uncomment the following line once the model is loaded
    # model.eval()
    test_loss, test_accuracy, test_f1, test_precision, test_recall = 0, 0, 0, 0, 0  # Initialize test metrics
    with torch.no_grad():
        # TODO: Implement test evaluation
        # Example:
        # for batch in test_dataloader:
        #     batch = [b.to(device) for b in batch]
        #     inputs, masks, labels = batch
        #     outputs = model(inputs, attention_mask=masks, labels=labels)
        #     test_loss += outputs.loss.item()
        #     TODO: Compute F1, precision, and recall for test evaluation
        #     preds = torch.argmax(outputs.logits, dim=1)
        #     test_accuracy += accuracy_score(labels.cpu(), preds.cpu())
        #     test_f1 += f1_score(labels.cpu(), preds.cpu(), average='weighted')
        #     test_precision += precision_score(labels.cpu(), preds.cpu(), average='weighted')
        #     test_recall += recall_score(labels.cpu(), preds.cpu(), average='weighted')
        #
        # test_loss /= len(test_dataloader)  # Average test loss
        # test_accuracy /= len(test_dataloader)  # Average accuracy
        # test_f1 /= len(test_dataloader)  # Average F1 score
        # test_precision /= len(test_dataloader)  # Average precision
        # test_recall /= len(test_dataloader)  # Average recall
        
        # print(f"Test Metrics: Loss: {test_loss}, Accuracy: {test_accuracy}, F1: {test_f1}, Precision: {test_precision}, Recall: {test_recall}")

        pass

    # TODO: Compute average test loss, accuracy, F1, precision, and recall
    # Example:
    # test_loss /= len(test_dataloader)
    # test_accuracy /= len(test_dataloader)
    # test_f1 /= len(test_dataloader)
    # test_precision /= len(test_dataloader)
    # test_recall /= len(test_dataloader)
    # print(f"Test Metrics: Loss: {test_loss}, Accuracy: {test_accuracy}, F1: {test_f1}, Precision: {test_precision}, Recall: {test_recall}")

    # Save model
    if save_model_path:
        # Example:
        # model.save_pretrained(save_model_path)
        # tokenizer.save_pretrained(save_model_path)
        pass

    return [seed, feature, test_loss, test_accuracy, test_f1, test_precision, test_recall]


if __name__ == "__main__":
    # usage
    gpu_numbers = None # Depending on the number of GPUs available. 0 for single GPU, "0,1" for multiple GPUs, None for CPU
    seed = None # TODO: set the seed for reproducibility
    save_model_path = "./models/"
    features = None # TODO: Add the list of features to fine-tune for (e.g. (sentiment_label, time_label, certain_label))

    for feature in features:
        print(f"Fine-tuning for feature: {feature}")
        results = fine_tune_plm(gpu_numbers=gpu_numbers, seed=seed, feature=feature, save_model_path=save_model_path + feature)
        print(f"Results for {feature}: {results}")
