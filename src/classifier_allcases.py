from typing import List
from utils import *
import torch
import torch.nn as nn
from transformers import RobertaTokenizer
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from transformers.optimization import get_linear_schedule_with_warmup
#from model import SentimentClassifier


class Classifier:
    """
    The Classifier: complete the definition of this class template by providing a constructor (i.e. the
    __init__() function) and the 2 methods train() and predict() below. Please donot change
     """

    ############################################# comp
    def train(self, train_filename: str, dev_filename: str, device: torch.device):
        """
        Trains the classifier model on the training set stored in file trainfile
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
          - DO NOT USE THE DEV DATA AS TRAINING EXAMPLES, YOU CAN USE THEM ONLY FOR THE OPTIMIZATION
         OF MODEL HYPERPARAMETERS
        """
        # Read in data
        train_data = pd.read_csv(train_filename,sep="\t",header = None)
        train_data.columns = ['polarity', 'aspect_category', 'target', 'index', 'sentence']

        if dev_filename is not None:
            dev_data = pd.read_csv(dev_filename,sep="\t",header = None)
            dev_data.columns = ['polarity', 'aspect_category', 'target', 'index', 'sentence']

        # Preprocess the data
        # Split the sentence
        train_data['sentence_left'] = train_data.apply(lambda row:rl_split(row['sentence'],row['index'])[0], axis = 1)
        train_data['sentence_right'] = train_data.apply(lambda row:rl_split(row['sentence'],row['index'])[1], axis = 1)
        train_data['aspect_category_processed'] = train_data['aspect_category'].apply(lambda x: transform_category(x))
        dev_data['sentence_left'] = dev_data.apply(lambda row:rl_split(row['sentence'],row['index'])[0], axis = 1)
        dev_data['sentence_right'] = dev_data.apply(lambda row:rl_split(row['sentence'],row['index'])[1], axis = 1)
        dev_data['aspect_category_processed'] = dev_data['aspect_category'].apply(lambda x: transform_category(x))

        # Label encoding
        train_data['polarity'] = train_data['polarity'].apply(lambda x:labelencoding(x))
        dev_data['polarity'] = dev_data['polarity'].apply(lambda x:labelencoding(x))

        self.n_classes = len(train_data['polarity'].unique()) #was train_filename before
        
        # Set the batch size
        self.batchsize = 32

        # Define Roberta tokenizer and model
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base') 
        self.model = SentimentClassifier(self.n_classes)
        self.model = self.model.to(device)

        self.max_length = max(sequence_length(train_data, self.tokenizer),sequence_length(dev_data, self.tokenizer))

        # Tokenize input data and create PyTorch DataLoader
        #X = train_data.apply(tokenize_text, axis=1)
        X = train_data.apply(lambda row: tokenize_text(row, self.tokenizer, self.max_length),axis = 1)
        input_ids = torch.cat([X[i]['input_ids'] for i in range(len(X))], dim=0)
        attention_masks = torch.cat([X[i]['attention_mask'] for i in range(len(X))], dim=0)
        y = torch.tensor(train_data['polarity'].values)

        # Tokenize test input data and create PyTorch DataLoader
        X_test = dev_data.apply(lambda row: tokenize_text(row, self.tokenizer, self.max_length),axis = 1)
        input_ids_test = torch.cat([X_test[i]['input_ids'] for i in range(len(X_test))], dim=0)
        attention_masks_test = torch.cat([X_test[i]['attention_mask'] for i in range(len(X_test))], dim=0)
        y_test = torch.tensor(dev_data['polarity'].values)
        test_dataset = TensorDataset(input_ids_test, attention_masks_test, y_test)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batchsize)

        # Set if you want to validate
        if dev_filename is not None:
            self.val = False
        else:
            self.val = True

        if self.val==True:
            # Split the dataset into training and validation sets
            train_X, val_X, train_y, val_y = train_test_split(input_ids, y, test_size=0.3, random_state=42)
            train_attention_masks, val_attention_masks, _, _ = train_test_split(attention_masks, y, test_size=0.3, random_state=42)

            # Create PyTorch DataLoader for the training and validation sets
            train_dataset = TensorDataset(train_X, train_attention_masks, train_y)
            train_dataloader = DataLoader(train_dataset, batch_size=self.batchsize, shuffle=True)

            val_dataset = TensorDataset(val_X, val_attention_masks, val_y)
            val_dataloader = DataLoader(val_dataset, batch_size=self.batchsize, shuffle=False)
        else:
            train_dataset = TensorDataset(input_ids, attention_masks, y)
            train_dataloader = DataLoader(train_dataset, batch_size=self.batchsize)
        


        # Train Roberta classifier
        self.epochs = 15
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #device='cpu'
        self.model.to(device)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5) #, weight_decay=0.001
        # Initialize GradScaler for mixed precision training
        if device!= 'cpu':
            from torch.cuda.amp import GradScaler, autocast
            scaler = GradScaler()
        else:
            autocast = lambda x: x  # Define a dummy autocast context manager for CPU

        # Add gradient accumulation steps
        gradient_accumulation_steps = 4

        # Define the total number of training steps and the number of warmup steps
        total_steps = len(train_dataloader) * self.epochs
        warmup_steps = 0 #int(total_steps * 0.1)
        # Create the learning rate scheduler using the warmup steps
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

        class_weights = np.array([1.43, 25.00, 3.85]) # Alleviate the imbalanced polarity dataset problem
        class_weights = torch.FloatTensor(class_weights).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        self.loss_list = []
        best_acc = 0

        for epoch in range(self.epochs):
            self.model.train()
            losses = []
            for i, batch in enumerate(train_dataloader):
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                labels = batch[2].to(device)

                # Enable mixed precision using autocast
                if device != 'cpu':
                    with autocast():
                        outputs = self.model(input_ids, attention_mask)
                        loss = loss_fn(outputs, labels)
                else:
                    outputs = self.model(input_ids, attention_mask)
                    loss = loss_fn(outputs, labels)

                if device != 'cpu':
                    # Scale the loss for mixed precision training
                    scaler.scale(loss).backward()

                    # Accumulate gradients and update weights every gradient_accumulation_steps
                    if (i + 1) % gradient_accumulation_steps == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                        scheduler.step()
                else:
                    loss.backward()
                    if (i + 1) % gradient_accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()

                losses.append(loss.item() * gradient_accumulation_steps)  # Correct the loss value
        
            # Calculate training accuracy
            self.model.eval()
            with torch.no_grad():
                train_preds = []
                train_labels = []
                for batch in train_dataloader:
                    input_ids = batch[0].to(device)
                    attention_mask = batch[1].to(device)
                    labels = batch[2].to(device)
                    outputs = self.model(input_ids, attention_mask)
                    _, predicted = torch.max(outputs, 1)
                    train_preds.extend(predicted.tolist())
                    train_labels.extend(labels.tolist())
                train_acc = np.mean(np.array(train_preds) == np.array(train_labels))
                
            if self.val == True:
                # Calculate validation accuracy
                with torch.no_grad():
                    val_preds = []
                    val_labels = []
                    for batch in val_dataloader:
                        input_ids = batch[0].to(device)
                        attention_mask = batch[1].to(device)
                        labels = batch[2].to(device)
                        outputs = self.model(input_ids, attention_mask)
                        _, predicted = torch.max(outputs, 1)
                        val_preds.extend(predicted.tolist())
                        val_labels.extend(labels.tolist())
                    val_acc = np.mean(np.array(val_preds) == np.array(val_labels))
                    if val_acc > best_acc:
                        torch.save(self.model, 'model_best.pt')

                print(f"Epoch {epoch}, Training Loss: {np.mean(losses)}, Training Accuracy: {train_acc}, Validation Accuracy: {val_acc}")
            else:
                # Calculate devdata accuracy
                with torch.no_grad():
                    test_preds = []
                    test_labels = []
                    for batch in test_dataloader:
                        input_ids = batch[0].to(device)
                        attention_mask = batch[1].to(device)
                        labels = batch[2].to(device)
                        outputs = self.model(input_ids, attention_mask)
                        _, predicted = torch.max(outputs, 1)
                        test_preds.extend(predicted.tolist())
                        test_labels.extend(labels.tolist())
                    test_acc = np.mean(np.array(test_preds) == np.array(test_labels))
                    if test_acc > best_acc:
                        torch.save(self.model, 'model_best.pt')
                        best_acc = test_acc
                print(f"Epoch {epoch}, Training Loss: {np.mean(losses)}, Training Accuracy: {train_acc}, Dev Accuracy: {test_acc}")
            self.loss_list.append(np.mean(losses))
            #torch.save(self.model, 'model_{}.pt'.format(epoch))
        # Load the best model(based on accuracy)
        self.model = torch.load('model_best.pt')


    def predict(self, data_filename: str, device: torch.device) -> List[str]:
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
        """

        data = pd.read_csv(data_filename,sep="\t",header = None)
        data.columns = ['polarity', 'aspect_category', 'target', 'index', 'sentence']
        
        # data preprocess
        data['sentence_left'] = data.apply(lambda row:rl_split(row['sentence'],row['index'])[0], axis = 1)
        data['sentence_right'] = data.apply(lambda row:rl_split(row['sentence'],row['index'])[1], axis = 1)
        data['aspect_category_processed'] = data['aspect_category'].apply(lambda x: transform_category(x))

        # Label encoding
        data['polarity'] = data['polarity'].apply(lambda x:labelencoding(x))

        self.max_length = sequence_length(data, self.tokenizer)

        # Tokenize test input data and create PyTorch DataLoader
        X = data.apply(lambda row: tokenize_text(row, self.tokenizer, self.max_length),axis = 1)
        input_ids = torch.cat([X[i]['input_ids'] for i in range(len(X))], dim=0)
        attention_masks = torch.cat([X[i]['attention_mask'] for i in range(len(X))], dim=0)
        y = torch.tensor(data['polarity'].values)
        dataset = TensorDataset(input_ids, attention_masks, y)
        dataloader = DataLoader(dataset, batch_size=self.batchsize)

        # Evaluate BERT classifier on test data
        self.model.eval()
        with torch.no_grad():
            y_pred = []
            for batch in dataloader:
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                labels = batch[2].to(device)
                outputs = self.model(input_ids, attention_mask=attention_mask)
                pred = outputs.argmax(dim=1)
                y_pred.extend(pred.cpu().numpy())

        output = []
        # converting integer labels into named labels
        for label in y_pred:
            if label == 0:
                output.append('positive')
            
            elif label == 1:
                output.append('neutral')
                
            elif label == 2:
                output.append('negative')

        return np.array(output)
