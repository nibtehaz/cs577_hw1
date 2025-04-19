class BertLinearProbing(nn.Module):
    def __init__(self, num_labels, n_epochs=10, val_data=None):
        super().__init__()
        self.num_labels = num_labels
        self.backbone = BertModel.from_pretrained("bert-base-uncased")

        self.attn = nn.MultiheadAttention(embed_dim=768,num_heads=24,batch_first=True)
        self.q_mpr = nn.Linear(768, 768)
        self.k_mpr = nn.Linear(768, 768)
        self.v_mpr = nn.Linear(768, 768)

        self.out = nn.Linear(768, self.num_labels)
        
        self.n_epochs = n_epochs
        self.val_dataset = val_data

        for param in self.backbone.encoder.layer[6:].parameters():
            param.requires_grad = True

    def forward(self, input_ids,atn_msk):
        # Perform one-hot encoding of input_ids
        with torch.no_grad():
            x = self.backbone(input_ids,attention_mask=atn_msk)['last_hidden_state']

        x = self.attn(,need_weights=False)
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
        
        # Training loop
        
        
        for epc in range(self.n_epochs):
            total_loss = 0
            #YP = []
            #YT = []
            total_f1 = []

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

                    #YP.append(yp)
                    #YT.append(yt)
                    total_f1.append(f1_score(yt, yp, labels = np.arange(13),average='macro'))


                 
            print(f'Epoch {epc+1} : Training Loss : {total_loss}, Macro F1 : {np.mean(total_f1)}')

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

                            YP.append(yp)
                            YT.append(yt)
                            total_f1.append(f1_score(yt, yp, labels = np.arange(13),average='macro'))


                        
                    print(f'Epoch {epc+1} : Val Loss : {total_loss}, Macro F1 : {np.mean(total_f1)}')



        return self, YP,YT

    def predict(self, dataset, batch_size=16) -> list[list[int]]:
        """
        Inference logic for NER task
        """
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)
        
        predictions = []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        self.eval()
        
        with torch.no_grad():
            # Note: do not use gold labels for the final inference, we will remove it while testing            
            for input_ids, _, valid_mask, atn_msk in tqdm(dataloader):
                input_ids, atn_msk = input_ids.to(device), atn_msk.to(device)                

                logits = self(input_ids,atn_msk)

                logits = torch.argmax(logits,dim=-1)
                        
                for i2 in range(len(valid_mask)):
                    yp = []
                    for j2 in range(len(valid_mask[i2])):
                        if valid_mask[i2][j2]==1:
                            yp.append(logits[i2][j2].item())                            

                    predictions.append(yp)

        return predictions

    
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





class BertLinearProbing(nn.Module):
    def __init__(self, num_labels, n_epochs=10, val_data=None):
        super().__init__()
        self.num_labels = num_labels
        self.backbone = BertModel.from_pretrained("bert-base-uncased")

        self.trnsfmr_lyr = nn.TransformerEncoderLayer(d_model=768, nhead=24, batch_first=True)

        self.out = nn.Linear(768, self.num_labels)
        
        self.n_epochs = n_epochs
        self.val_dataset = val_data

        for param in self.backbone.encoder.layer[:8].parameters():
            param.requires_grad = True
        for param in self.backbone.encoder.layer[8:].parameters():
            param.requires_grad = False

    def forward(self, input_ids,atn_msk):
        # Perform one-hot encoding of input_ids
        #with torch.no_grad():
        x = self.backbone(input_ids,attention_mask=atn_msk)['last_hidden_state']

        #x = self.trnsfmr_lyr(x,src_key_padding_mask=input_ids.eq(0))
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
        
        # Training loop
        
        
        for epc in range(self.n_epochs):
            total_loss = 0
            #YP = []
            #YT = []
            total_f1 = []

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

                    #YP.append(yp)
                    #YT.append(yt)
                    total_f1.append(f1_score(yt, yp, labels = np.arange(13),average='macro'))


                 
            print(f'Epoch {epc+1} : Training Loss : {total_loss}, Macro F1 : {np.mean(total_f1)}')

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

                            YP.append(yp)
                            YT.append(yt)
                            total_f1.append(f1_score(yt, yp, labels = np.arange(13),average='macro'))


                        
                    print(f'Epoch {epc+1} : Val Loss : {total_loss}, Macro F1 : {np.mean(total_f1)}')



        return self, YP,YT

    def predict(self, dataset, batch_size=16) -> list[list[int]]:
        """
        Inference logic for NER task
        """
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)
        
        predictions = []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        self.eval()
        
        with torch.no_grad():
            # Note: do not use gold labels for the final inference, we will remove it while testing            
            for input_ids, _, valid_mask, atn_msk in tqdm(dataloader):
                input_ids, atn_msk = input_ids.to(device), atn_msk.to(device)                

                logits = self(input_ids,atn_msk)

                logits = torch.argmax(logits,dim=-1)
                        
                for i2 in range(len(valid_mask)):
                    yp = []
                    for j2 in range(len(valid_mask[i2])):
                        if valid_mask[i2][j2]==1:
                            yp.append(logits[i2][j2].item())                            

                    predictions.append(yp)

        return predictions

    
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

