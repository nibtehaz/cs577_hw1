import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder



'''
This is a file that contains the code for importing the dataset and training a model on it. Your task is to replace the model here and come up with your own
model. We will train your model on the training set using the train() method you write, then test it on the test set.
'''
class WNUTDataset(Dataset):
    def __init__(self, data, label_encoder):
        self.data = data
        self.label_encoder = label_encoder
        
        # Preprocess data
        self.sentences = [example['tokens'] for example in self.data]
        self.labels = [example['ner_tags'] for example in self.data]
        
        # Tokenize (basic) and encode labels
        self.tokenized_inputs = [[word.lower() for word in sentence] for sentence in self.sentences]
        # Label encoding tokenizer - assign a number to each word in the dataset vocabulary.
        self.encoded_labels = [self.label_encoder.transform(labels) for labels in self.labels]
        
        # Build vocabulary
        self.word_to_idx = {word: idx + 1 for idx, word in enumerate({word for sentence in self.tokenized_inputs for word in sentence})}
        self.word_to_idx['<PAD>'] = 0  # Padding token

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        tokenized_sentence = self.tokenized_inputs[idx]
        input_ids = torch.tensor([self.word_to_idx.get(word, 0) for word in tokenized_sentence], dtype=torch.long)
        labels = torch.tensor(self.encoded_labels[idx], dtype=torch.long)
        return input_ids, labels

class SimpleLogisticRegressionModel(nn.Module):
    def __init__(self, vocab_size, num_labels):
        super(SimpleLogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(vocab_size, num_labels)

    def forward(self, input_ids):
        # Perform one-hot encoding of input_ids
        one_hot_inputs = F.one_hot(input_ids, num_classes=self.linear.in_features).float()
        logits = self.linear(one_hot_inputs)
        return logits

    def train(self, dataset, batch_size=1):
        # Prepare DataLoader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
        
        # Optimizer and Loss function - feel free to change
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        
        # Training loop
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        
        total_loss = 0
        for input_ids, labels in dataloader:
            input_ids, labels = input_ids.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = self(input_ids)
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return self

# This function is meant for padding the dataset's examples so that every input is the same length
def pad_collate_fn(batch):
    input_ids, labels = zip(*batch)
    max_len = max(len(ids) for ids in input_ids)
    padded_inputs = [F.pad(ids, (0, max_len - len(ids)), value=0) for ids in input_ids]
    padded_labels = [F.pad(label, (0, max_len - len(label)), value=-1) for label in labels]
    return torch.stack(padded_inputs), torch.stack(padded_labels)

def main():
    # Load dataset - this will use a couple megabytes of space on your machine
    dataset = load_dataset('leondz/wnut_17',trust_remote_code=True)

    # Label encoder
    label_encoder = LabelEncoder()

    # Fit label encoder to all labels in the dataset
    all_labels = [label for example in dataset['train'] for label in example['ner_tags']]
    label_encoder.fit(all_labels)

    # Prepare datasets
    train_dataset = WNUTDataset(dataset['train'], label_encoder)
    val_dataset = WNUTDataset(dataset['validation'], label_encoder)

    # Model
    vocab_size = len(train_dataset.word_to_idx) + 1
    num_labels = len(label_encoder.classes_)
    model = SimpleLogisticRegressionModel(vocab_size, num_labels)

    # Train the model
    model.train(train_dataset, batch_size=16)

if __name__ == "__main__":
    main()

