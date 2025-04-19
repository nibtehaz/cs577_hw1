import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report
from torchcrf import CRF


class WNUTDataset(Dataset):
    def __init__(self, data, label_encoder):
        self.data = data
        self.label_encoder = label_encoder
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # Preprocess data
        self.sentences = [example['tokens'] for example in self.data]
        self.labels = [example['ner_tags'] for example in self.data]
        self.num_classes = max(label_encoder.classes_)
        
        # Tokenize (basic) and encode labels
        self.tokenized_inputs = [[word.lower() for word in sentence] for sentence in self.sentences]
        # Label encoding tokenizer - assign a number to each word in the dataset vocabulary.
        self.encoded_labels = [self.label_encoder.transform(labels) for labels in self.labels]
        
        # Build vocabulary
        # using bert vocab

    def __len__(self):
        return len(self.sentences)

    def adjust_token_and_label(self,tokens_by_wrd,wrd_lbls):

        tokens = []
        valid_mask = []
        labels = []

        for i in range(len(tokens_by_wrd)):           
            if len(tokens_by_wrd[i]) < 3:
                tokens.append(tokens_by_wrd[i][0])
                valid_mask.append(1)
                labels.append(wrd_lbls[i])

            for j in range(1,len(tokens_by_wrd[i])-1):
                tokens.append(tokens_by_wrd[i][j])
                valid_mask.append(1 if j==1 else 0)
                labels.append(wrd_lbls[i] if j==1 else self.num_classes+1)

        return torch.tensor(tokens, dtype=torch.long), torch.tensor(labels, dtype=torch.long), torch.tensor(valid_mask, dtype=torch.int)

    def __getitem__(self, idx):
        
        tokenized_sentence = self.tokenized_inputs[idx]
        
        bert_tokens_by_wrd = self.tokenizer(tokenized_sentence)['input_ids']
        wrd_lbls = torch.tensor(self.encoded_labels[idx], dtype=torch.long)

        tokens, labels, valid_mask = self.adjust_token_and_label(bert_tokens_by_wrd,wrd_lbls)

        return tokens, labels, valid_mask



class NERModel(nn.Module):
    def __init__(self, num_labels=14, n_epochs=10, val_data=None):
        super().__init__()
        self.num_labels = num_labels
        self.backbone = BertModel.from_pretrained("bert-base-uncased")

        self.out = nn.Linear(768, self.num_labels)
        
        self.n_epochs = n_epochs
        self.val_dataset = val_data

        self.pst_prcssing = Postprorcessing(num_labels,3,val_data)


        
    def forward(self, input_ids,atn_msk):
        # Perform one-hot encoding of input_ids
        with torch.no_grad():
            x = self.backbone(input_ids,attention_mask=atn_msk)['last_hidden_state']

        logits = self.out(x)
        return logits


    def train_model(self, train_dataset, batch_size=16):
        # Prepare DataLoader

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)

        if self.val_dataset is not None:
            val_dataloader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)


        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        
        
        # Optimizer and Loss function - feel free to change
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([0.1]+[1.0]*12+[0.0]).to(device), ignore_index=13)
        
        best_loss = None 
        best_model = None
        # Training loop
        
        
        for epc in range(self.n_epochs):
            total_loss = 0
            YP = []
            YT = []            

            self.train()

            for input_ids, labels, valid_mask, atn_msk in tqdm(train_dataloader):

                input_ids, labels, atn_msk = input_ids.to(device), labels.to(device), atn_msk.to(device)                

                optimizer.zero_grad()

                logits = self(input_ids,atn_msk)
                
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                loss.backward()
                
                optimizer.step()

                total_loss += loss.item()

                logits = torch.argmax(logits,dim=-1)
                
                for i2 in range(len(valid_mask)):
                    yp = []
                    yt = []
                    for j2 in range(len(valid_mask[i2])):
                        if valid_mask[i2][j2]==1:
                            yp.append(logits[i2][j2].item())
                            yt.append(labels[i2][j2].item())

                    YP.extend(yp)
                    YT.extend(yt)
                    

            f1_mcro_scr = f1_score(YT, YP, average='macro')
            print(f'Epoch [{epc+1}/10] : Training Loss : {total_loss}, Macro F1 : {f1_mcro_scr}')

            if self.val_dataset is not None:
                total_loss = 0
                YP = []
                YT = []
                total_f1 = []

                self.eval()

                with torch.no_grad():
                    for input_ids, labels, valid_mask, atn_msk in tqdm(val_dataloader):

                        input_ids, labels, atn_msk = input_ids.to(device), labels.to(device), atn_msk.to(device)                

                        logits = self(input_ids,atn_msk)
                        
                        loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                        
                        total_loss += loss.item()

                        logits = torch.argmax(logits,dim=-1)
                        
                        for i2 in range(len(valid_mask)):
                            yp = []
                            yt = []
                            for j2 in range(len(valid_mask[i2])):
                                if valid_mask[i2][j2]==1:
                                    yp.append(logits[i2][j2].item())
                                    yt.append(labels[i2][j2].item())

                            YP.extend(yp)
                            YT.extend(yt)
                            
                    if (best_loss is None) or (best_loss > total_loss):
                        best_loss = total_loss
                        best_model = self
                    f1_mcro_scr = f1_score(YT, YP, average='macro')    
                    print(f'Epoch [{epc+1}/10] : Val Loss : {total_loss}, Macro F1 : {f1_mcro_scr}')
        
        if (best_model is None):
            best_model = self

        best_model.pst_prcssing = best_model.pst_prcssing.train_model(train_dataset, best_model, batch_size=16)

        
        return best_model
        

    def predict(self, dataset, batch_size=16) -> list[list[int]]:
        """
        Inference logic for NER task
        """
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)
        use_crf = True
        predictions = []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        self.eval()
        
        with torch.no_grad():
            # Note: do not use gold labels for the final inference, we will remove it while testing            
            for input_ids, _, valid_mask, atn_msk in tqdm(dataloader):
                input_ids, atn_msk = input_ids.to(device), atn_msk.to(device)                

                logits = self(input_ids,atn_msk)

                ###logits = torch.argmax(logits,dim=-1)
                if use_crf:
                    logits = self.pst_prcssing.predict(logits,atn_msk)

                else:
                    logits = torch.argmax(logits,dim=-1).tolist()
                        
                for i2 in range(len(valid_mask)):
                    yp = []
                    for j2 in range(len(valid_mask[i2])):
                        if valid_mask[i2][j2]==1:
                            yp.append(logits[i2][j2])                            

                    predictions.append(yp)

        return predictions


class Postprorcessing(nn.Module):
    def __init__(self, num_labels, n_epochs=10, val_data=None):
        super().__init__()
        self.crf_mdl = CRF(num_labels, batch_first=True)
        self.val_dataset = val_data
        self.n_epochs = n_epochs

    def forward(self, predictions, labels, masks):
        return -self.crf_mdl(predictions, labels, mask=masks)

    def train_model(self, train_dataset, trained_mdl, batch_size=16):

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)

        if self.val_dataset is not None:
            val_dataloader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)


        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        trained_mdl.to(device)
        
        # Optimizer and Loss function - feel free to change
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        
        best_loss = None 
        best_model = None
        # Training loop
        
        
        for epc in range(self.n_epochs):
            total_loss = 0
            
            self.train()

            for input_ids, labels, valid_mask, atn_msk in tqdm(train_dataloader):

                input_ids, labels, atn_msk = input_ids.to(device), labels.to(device), atn_msk.to(device)                

                optimizer.zero_grad()
                
                with torch.no_grad():
                    logits = trained_mdl(input_ids,atn_msk)
                
                loss = self(logits,labels,atn_msk)
                
                loss.backward()
                
                optimizer.step()

                total_loss += loss.item()


                 
            tqdm.write(f'CRF Epoch [{epc+1}/3] : Training Loss : {total_loss}')

            if self.val_dataset is not None:
                total_loss = 0                

                self.eval()

                with torch.no_grad():
                    for input_ids, labels, valid_mask, atn_msk in tqdm(val_dataloader):

                        input_ids, labels, atn_msk = input_ids.to(device), labels.to(device), atn_msk.to(device)                

                        with torch.no_grad():
                            logits = trained_mdl(input_ids,atn_msk)
                        
                        loss = self(logits,labels,atn_msk)
                        
                        total_loss += loss.item()


                    if (best_loss is None) or (best_loss > total_loss):
                        best_loss = total_loss
                        best_model = self
                        
                    tqdm.write(f'CRF Epoch [{epc+1}/3] : Val Loss : {total_loss}')

        if best_model is not None:
            return best_model
        else:
            return self


    def predict(self, predictions, masks):
        return self.crf_mdl.decode(predictions, mask=masks)



# This function is meant for padding the dataset's examples so that every input is the same length
def pad_collate_fn(batch):
    
    input_ids, labels, valid_masks = zip(*batch)
    
    max_len = max(len(ids) for ids in input_ids)

    padded_inputs = [F.pad(ids, (0, max_len - len(ids)), value=0) for ids in input_ids]
    padded_inputs = torch.stack(padded_inputs)

    if labels is not None:    
        padded_labels = [F.pad(label, (0, max_len - len(label)), value=13) for label in labels]
        padded_labels = torch.stack(padded_labels)
    else:
        padded_labels = None
    
    padded_valid_masks = [F.pad(valid_mask, (0, max_len - len(valid_mask)), value=0) for valid_mask in valid_masks]
    padded_valid_masks = torch.stack(padded_valid_masks)

    attention_mask = padded_inputs.not_equal(0).clone().detach()

    return padded_inputs, padded_labels, padded_valid_masks, attention_mask



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
    test_dataset = WNUTDataset(dataset['test'], label_encoder)

    # Model
    model = NERModel()
    # Train the model
    model = model.train_model(train_dataset, batch_size=16)
    
    # Predicting the test data
    predictions = model.predict(test_dataset, batch_size=16)
    
    # compute metrics
    true_labels = [label for sentence in dataset["test"] for label in sentence["ner_tags"]]

    predicted_labels = [label for sentence in predictions for label in sentence]

    macro_f1 = f1_score(true_labels, predicted_labels, average="macro")

    print(f"Macro F1 Score: {macro_f1:.4f}")

    print(classification_report(true_labels, predicted_labels))


if __name__ == "__main__":
    main()