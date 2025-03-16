################################################################################
#Image-Text Matching Classifier: Baseline System for Visual Question Answering
#Version 1.0, Main Functionality in Tensorflow Tested with COCO Data 
#Version 1.2, Extended Functionality for Flickr Data
#Version 1.3, Ported to PyTorch and Tested with Visual7w Data
################################################################################

import os
import time
import pickle
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.models import vit_b_32  

#Custom Dataset
class ITM_Dataset(Dataset):
    def __init__(self, images_path, data_file, sentence_embeddings, data_split, train_ratio=1.0):
        self.images_path = images_path
        self.data_file = data_file
        self.sentence_embeddings = sentence_embeddings
        self.data_split = data_split.lower()
        self.train_ratio = train_ratio if self.data_split == "train" else 1.0

        self.image_data = []
        self.question_data = []
        self.answer_data = []
        self.question_embeddings_data = []
        self.answer_embeddings_data = []
        self.label_data = []
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            #Standard for Pretrained Models on ImageNet
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
        ])

        self.load_data()

    def load_data(self):
        print("LOADING data from "+str(self.data_file))
        print("=========================================")

        random.seed(42)

        with open(self.data_file) as f:
            lines = f.readlines()

            #Apply Train_Ratio Only for Training Data
            if self.data_split == "train":
                #Shuffle Before Selecting
                random.shuffle(lines)  
                num_samples = int(len(lines) * self.train_ratio)
                lines = lines[:num_samples]

            for line in lines:
                line = line.rstrip("\n")
                img_name, text, raw_label = line.split("\t")  
                img_path = os.path.join(self.images_path, img_name.strip())

                question_answer_text = text.split("?")
                question_text = question_answer_text[0].strip() + '?'
                answer_text = question_answer_text[1].strip()

                #Get Binary Labels from match/no-match Answers
                label = 1 if raw_label == "match" else 0
                self.image_data.append(img_path)
                self.question_data.append(question_text)
                self.answer_data.append(answer_text)
                self.question_embeddings_data.append(self.sentence_embeddings[question_text])
                self.answer_embeddings_data.append(self.sentence_embeddings[answer_text])
                self.label_data.append(label)

        print("|image_data|="+str(len(self.image_data)))
        print("|question_data|="+str(len(self.question_data)))
        print("|answer_data|="+str(len(self.answer_data)))
        print("done loading data...")

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        img_path = self.image_data[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)  
        question_embedding = torch.tensor(self.question_embeddings_data[idx], dtype=torch.float32)
        answer_embedding = torch.tensor(self.answer_embeddings_data[idx], dtype=torch.float32)
        label = torch.tensor(self.label_data[idx], dtype=torch.long)
        return img, question_embedding, answer_embedding, label

#Load Sentence Embeddings from Existing File -- Generated a Priori
def load_sentence_embeddings(file_path):
    print("READING sentence embeddings...")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

#Pre-trained ViT Model
class Transformer_VisionEncoder(nn.Module):
    def __init__(self, pretrained=None):
        super(Transformer_VisionEncoder, self).__init__()

        if pretrained:
            self.vision_model = vit_b_32(weights="IMAGENET1K_V1")
            #Freeze All Layers Initially
            for param in self.vision_model.parameters():
                param.requires_grad = False

            #Unfreeze Last two Layers
            for param in list(self.vision_model.heads.parameters())[-2:]:
                param.requires_grad = True
        else:
            #Initialize Without Pretrained Weights
            self.vision_model = vit_b_32(weights=None)  
	
        #Get Feature Size After Initialising Model
        self.num_features = self.vision_model.heads[0].in_features

        #Remove Original Classification Head
        self.vision_model.heads = nn.Identity()

    def forward(self, x):
        #Shape should be (batch_size, num_features)
        features = self.vision_model(x)  
        return features

#CNN with Attention Mechanism
class CNN_Attention(nn.Module):
    def __init__(self, pretrained=None):
        super(CNN_Attention, self).__init__()

        #Use ResNet18 as Base CNN
        self.cnn = models.resnet18(pretrained=pretrained)
        if pretrained:
            #Freeze All Layers Initially
            for param in self.cnn.parameters():
                param.requires_grad = False
            #Unfreeze Last two Layers
            for param in list(self.cnn.children())[-2:]:
                for p in param.parameters():
                    p.requires_grad = True
        else:
            for param in self.cnn.parameters():
                param.requires_grad = True

        #Remove Final Classification Layer
        self.cnn.fc = nn.Identity()

        #Attention Mechanism
        self.attention = nn.Sequential(
            #ResNet18 Outputs 512 Features
            nn.Linear(512, 256),  
            nn.Tanh(),
            nn.Linear(256, 1),
            nn.Softmax(dim=1))
        
        #Fully Connected Layer for Final Output
        self.fc = nn.Linear(512, 128)

    def forward(self, x):
        #Extract Features using CNN
        #Shape: (batch_size, 512)
        features = self.cnn(x)  

        #Apply Attention Mechanism
        #Shape: (batch_size, 1)
        attention_weights = self.attention(features)  
        #Shape: (batch_size, 512)
        attended_features = features * attention_weights  

        #Pass Through Final Fully Connected Layer
        #Shape: (batch_size, 128)
        output = self.fc(attended_features)  
        return output

#Image-Text Matching Model
class ITM_Model(nn.Module):
    def __init__(self, num_classes=2, ARCHITECTURE=None, PRETRAINED=None):
        print(f'BUILDING %s model, pretrained=%s' % (ARCHITECTURE, PRETRAINED))
        super(ITM_Model, self).__init__()
        self.ARCHITECTURE = ARCHITECTURE

        if self.ARCHITECTURE == "CNN":
            self.vision_model = models.resnet18(pretrained=PRETRAINED)
            if PRETRAINED:
			    #Freeze All Layers 
                for param in self.vision_model.parameters():
                    param.requires_grad = False
                #Unfreeze Last two Layers
                for param in list(self.vision_model.children())[-2:]:
                    for p in param.parameters():
                        p.requires_grad = True
            else:
                for param in self.vision_model.parameters():
                    param.requires_grad = True
                    #Change Output
            self.vision_model.fc = nn.Linear(self.vision_model.fc.in_features, 128) 

        elif self.ARCHITECTURE == "ViT":
            self.vision_model = Transformer_VisionEncoder(pretrained=PRETRAINED)
            #Reduce Features
            self.fc_vit = nn.Linear(self.vision_model.num_features, 128)  

        elif self.ARCHITECTURE == "CNN_Attention":
            self.vision_model = CNN_Attention(pretrained=PRETRAINED)
            #Adjust Feature Size
            self.fc_cnn_attention = nn.Linear(128, 128)  

        else:
            print("UNKNOWN neural architecture", ARCHITECTURE)
            exit(0)
        #Adjust Question Dimension
        self.question_embedding_layer = nn.Linear(768, 128)  
        #Adjust Answer Dimension 
        self.answer_embedding_layer = nn.Linear(768, 128)  
        #Concatenate Vision and Text Features
        self.fc = nn.Linear(128 + 128 + 128, num_classes)  

    def forward(self, img, question_embedding, answer_embedding):
        img_features = self.vision_model(img)
        if self.ARCHITECTURE == "ViT":
            #Use Custom Linear Layer for ViT
            img_features = self.fc_vit(img_features) 
        elif self.ARCHITECTURE == "CNN_Attention":
            #Use Custom Linear Layer for CNN_Attention
            img_features = self.fc_cnn_attention(img_features)  
        question_features = self.question_embedding_layer(question_embedding)
        answer_features = self.answer_embedding_layer(answer_embedding)
        combined_features = torch.cat((img_features, question_features, answer_features), dim=1)
        output = self.fc(combined_features)
        return output

def train_model(model, ARCHITECTURE, train_loader, criterion, optimiser, num_epochs=10):
    print(f'TRAINING %s model' % (ARCHITECTURE))
    model.train()
    
    #Track Overall Loss for Epoch
    for epoch in range(num_epochs):
        running_loss = 0.0
        total_batches = len(train_loader)
        #Record Start Time for Epoch
        start_time = time.time()  

        for batch_idx, (images, question_embeddings, answer_embeddings, labels) in enumerate(train_loader):
            #Move Images/Text/Labels to GPU (If Available)
            images = images.to(device)          
            question_embeddings = question_embeddings.to(device) 
            answer_embeddings = answer_embeddings.to(device)  
            labels = labels.to(device)

            #Forward Pass -- Given Input Data to Model
            outputs = model(images, question_embeddings, answer_embeddings)

            #Calculate Loss (Error)
            #Output should be Raw Logits
            loss = criterion(outputs, labels)  
			
            #Backward Pass -- Given Loss Above
            #Clear Gradients
            optimiser.zero_grad() 
            #Computes Gradient of Loss/Error
            loss.backward() 
            #Updates Parameters using Gradients
            optimiser.step() 
            running_loss += loss.item()

            #Print Progress every X Batches
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{total_batches}], Loss: {loss.item():.4f}')
        
        #Calculate Epoch Training Time
        #Time Taken for Epoch
        epoch_time = time.time() - start_time  

        #Print Average Loss and Training Time for Epoch
        avg_loss = running_loss / total_batches
        print(f'Epoch [{epoch + 1}/{num_epochs}] Average Loss: {avg_loss:.4f}, Time: {epoch_time:.2f} seconds')

    #Calculate Total Training Time
    #Total Time for All Epochs
    total_training_time = time.time() - start_time  
    print(f'Total Training Time: {total_training_time:.2f} seconds')

def evaluate_model(model, ARCHITECTURE, test_loader, device):
    print(f'EVALUATING %s model' % (ARCHITECTURE))
    model.eval()
    total_test_loss = 0
    all_labels = []
    all_predictions = []
    #Store Reciprocal Ranks for MRR Calculation
    reciprocal_ranks = []  
    start_time = time.time()

    with torch.no_grad():
        for images, question_embeddings, answer_embeddings, labels in test_loader:
            #Move Images/Text/Labels to GPU (if Available)
            images = images.to(device)          
            question_embeddings = question_embeddings.to(device) 
            answer_embeddings = answer_embeddings.to(device)  
            #Labels are Single Integers (0 or 1)
            labels = labels.to(device)  
			
            #Perform Forward Pass on Our Data
            outputs = model(images, question_embeddings, answer_embeddings)
			
            #Accumulate Loss on Test Data
            total_test_loss += criterion(outputs, labels)  

            #Apply Softmax to Get Probabilities
            #Use Softmax for Multi-Class Output
            predicted_probabilities = torch.softmax(outputs, dim=1)  
            #Get Predicted Class Index (0 or 1)
            predicted_class = predicted_probabilities.argmax(dim=1)  

            #Store Labels and Predictions for Later Analysis
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted_class.cpu().numpy())

            #Calculate MRR
            for i in range(len(labels)):
                #Get Predicted Probabilities for Current Question
                question_probs = predicted_probabilities[i].cpu().numpy()

                #Sort Candidate Answers by their Predicted Probabilities (Descending Order)
                ranked_indices = np.argsort(question_probs)[::-1]

                #Find Rank of Correct Answer
                correct_index = labels[i].item()
                #Add 1 because Ranks Start at 1
                rank = np.where(ranked_indices == correct_index)[0][0] + 1  

                #Calculate Reciprocal Rank
                reciprocal_rank = 1.0 / rank
                reciprocal_ranks.append(reciprocal_rank)

    #Convert to Numpy Arrays for Easier Calculations
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    #Calculate True Positives, True Negatives, False Positives, False Negatives
    #True Positives
    tp = np.sum((all_predictions == 1) & (all_labels == 1))  
    #True Negatives
    tn = np.sum((all_predictions == 0) & (all_labels == 0))  
    #False Positives
    fp = np.sum((all_predictions == 1) & (all_labels == 0))  
    #False Negatives
    fn = np.sum((all_predictions == 0) & (all_labels == 1))  

    #Calculate Sensitivity, Specificity, Balanced Accuracy, and F1-Score
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp)
    balanced_accuracy = (sensitivity + specificity) / 2.0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)

    #Calculate Mean Reciprocal Rank (MRR)
    mrr = np.mean(reciprocal_ranks)

    elapsed_time = time.time() - start_time
    print(f'Balanced Accuracy: {balanced_accuracy:.4f}, {elapsed_time:.2f} seconds')
    print(f'F1-Score: {f1_score}')
    print(f'Total Test Loss: {total_test_loss:.4f}')
    print(f'Mean Reciprocal Rank (MRR): {mrr:.4f}')

#Main Execution
if __name__ == '__main__':
    #Check GPU Availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    #Paths and Files
    IMAGES_PATH = "ITM_Classifier-baselines/visual7w-images"
    train_data_file = "ITM_Classifier-baselines/visual7w-text/v7w.TrainImages.itm.txt"
    dev_data_file = "ITM_Classifier-baselines/visual7w-text/v7w.DevImages.itm.txt"
    test_data_file = "ITM_Classifier-baselines/visual7w-text/v7w.TestImages.itm.txt"
    sentence_embeddings_file = "ITM_Classifier-baselines/v7w.sentence_embeddings-gtr-t5-large.pkl"
    sentence_embeddings = load_sentence_embeddings(sentence_embeddings_file)

    #Create Datasets and Loaders
    train_dataset = ITM_Dataset(IMAGES_PATH, train_data_file, sentence_embeddings, data_split="train", train_ratio=0.2)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    #Whole Test Data
    test_dataset = ITM_Dataset(IMAGES_PATH, test_data_file, sentence_embeddings, data_split="test")  
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)  

    #Create Model using One of Supported Architectures
    #Options: "CNN", "ViT", or "CNN_Attention"
    MODEL_ARCHITECTURE = "ViT"  
    USE_PRETRAINED_MODEL = True
    model = ITM_Model(num_classes=2, ARCHITECTURE=MODEL_ARCHITECTURE, PRETRAINED=USE_PRETRAINED_MODEL).to(device)
    print("\nModel Architecture:")
    print(model)

    #Print Parameters of Model Selected
    total_params = 0
    print("\nModel Trainable Parameters:")
    for name, param in model.named_parameters():
        #Print Trainable Parameters
        if param.requires_grad:  
            num_params = param.numel()  
            total_params += num_params
            print(f"{name}: {param.data.shape} | Number of parameters: {num_params}")
    print(f"\nTotal number of parameters in the model: {total_params}")
    print(f"\nUSE_PRETRAINED_MODEL={USE_PRETRAINED_MODEL}\n")

    #Define Loss Function and Optimiser 
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)

    #Train and Evaluate Model
    train_model(model, MODEL_ARCHITECTURE, train_loader, criterion, optimiser, num_epochs=10)
    evaluate_model(model, MODEL_ARCHITECTURE, test_loader, device)