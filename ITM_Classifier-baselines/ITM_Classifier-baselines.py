################################################################################
# Image-Text Matching Classifier: baseline system for visual question answering
# Version 1.0, main functionality in tensorflow tested with COCO data 
# Version 1.2, extended functionality for Flickr data
# Version 1.3, ported to pytorch and tested with visual7w data
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

# Custom Dataset
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
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard for pretrained models on ImageNet
        ])

        self.load_data()

    def load_data(self):
        print("LOADING data from "+str(self.data_file))
        print("=========================================")

        random.seed(42)

        with open(self.data_file) as f:
            lines = f.readlines()

            # Apply train_ratio only for training data
            if self.data_split == "train":
                random.shuffle(lines)  # Shuffle before selecting
                num_samples = int(len(lines) * self.train_ratio)
                lines = lines[:num_samples]

            for line in lines:
                line = line.rstrip("\n")
                img_name, text, raw_label = line.split("\t")  
                img_path = os.path.join(self.images_path, img_name.strip())

                question_answer_text = text.split("?")
                question_text = question_answer_text[0].strip() + '?'
                answer_text = question_answer_text[1].strip()

                # Get binary labels from match/no-match answers
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

# Load sentence embeddings from an existing file -- generated a priori
def load_sentence_embeddings(file_path):
    print("READING sentence embeddings...")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

# Pre-trained ViT model
class Transformer_VisionEncoder(nn.Module):
    def __init__(self, pretrained=None):
        super(Transformer_VisionEncoder, self).__init__()

        if pretrained:
            self.vision_model = vit_b_32(weights="IMAGENET1K_V1")
            # Freeze all layers initially
            for param in self.vision_model.parameters():
                param.requires_grad = False

            # Unfreeze the last two layers
            for param in list(self.vision_model.heads.parameters())[-2:]:
                param.requires_grad = True
        else:
            self.vision_model = vit_b_32(weights=None)  # Initialize without pretrained weights
	
        # Get feature size after initialising the model
        self.num_features = self.vision_model.heads[0].in_features

        # Remove original classification head
        self.vision_model.heads = nn.Identity()

    def forward(self, x):
        features = self.vision_model(x)  # Shape should be (batch_size, num_features)
        return features

# CNN with Attention Mechanism
class CNN_Attention(nn.Module):
    def __init__(self, pretrained=None):
        super(CNN_Attention, self).__init__()

        # Use ResNet18 as the base CNN
        self.cnn = models.resnet18(pretrained=pretrained)
        if pretrained:
            # Freeze all layers initially
            for param in self.cnn.parameters():
                param.requires_grad = False
            # Unfreeze the last two layers
            for param in list(self.cnn.children())[-2:]:
                for p in param.parameters():
                    p.requires_grad = True
        else:
            for param in self.cnn.parameters():
                param.requires_grad = True

        # Remove the final classification layer
        self.cnn.fc = nn.Identity()

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(512, 256),  # ResNet18 outputs 512 features
            nn.Tanh(),
            nn.Linear(256, 1),
            nn.Softmax(dim=1))
        
        # Fully connected layer for final output
        self.fc = nn.Linear(512, 128)

    def forward(self, x):
        # Extract features using CNN
        features = self.cnn(x)  # Shape: (batch_size, 512)

        # Apply attention mechanism
        attention_weights = self.attention(features)  # Shape: (batch_size, 1)
        attended_features = features * attention_weights  # Shape: (batch_size, 512)

        # Pass through final fully connected layer
        output = self.fc(attended_features)  # Shape: (batch_size, 128)
        return output

# Image-Text Matching Model
class ITM_Model(nn.Module):
    def __init__(self, num_classes=2, ARCHITECTURE=None, PRETRAINED=None):
        print(f'BUILDING %s model, pretrained=%s' % (ARCHITECTURE, PRETRAINED))
        super(ITM_Model, self).__init__()
        self.ARCHITECTURE = ARCHITECTURE

        if self.ARCHITECTURE == "CNN":
            self.vision_model = models.resnet18(pretrained=PRETRAINED)
            if PRETRAINED:
			    # Freeze all layers 
                for param in self.vision_model.parameters():
                    param.requires_grad = False
                # Unfreeze the last two layers
                for param in list(self.vision_model.children())[-2:]:
                    for p in param.parameters():
                        p.requires_grad = True
            else:
                for param in self.vision_model.parameters():
                    param.requires_grad = True
            self.vision_model.fc = nn.Linear(self.vision_model.fc.in_features, 128) # Change output

        elif self.ARCHITECTURE == "ViT":
            self.vision_model = Transformer_VisionEncoder(pretrained=PRETRAINED)
            self.fc_vit = nn.Linear(self.vision_model.num_features, 128)  # Reduce features

        elif self.ARCHITECTURE == "CNN_Attention":
            self.vision_model = CNN_Attention(pretrained=PRETRAINED)
            self.fc_cnn_attention = nn.Linear(128, 128)  # Adjust feature size

        else:
            print("UNKNOWN neural architecture", ARCHITECTURE)
            exit(0)

        self.question_embedding_layer = nn.Linear(768, 128)  # Adjust question dimension 
        self.answer_embedding_layer = nn.Linear(768, 128)  # Adjust answer dimension
        self.fc = nn.Linear(128 + 128 + 128, num_classes)  # Concatenate vision and text features

    def forward(self, img, question_embedding, answer_embedding):
        img_features = self.vision_model(img)
        if self.ARCHITECTURE == "ViT":
            img_features = self.fc_vit(img_features) # Use the custom linear layer for ViT
        elif self.ARCHITECTURE == "CNN_Attention":
            img_features = self.fc_cnn_attention(img_features)  # Use the custom linear layer for CNN_Attention
        question_features = self.question_embedding_layer(question_embedding)
        answer_features = self.answer_embedding_layer(answer_embedding)
        combined_features = torch.cat((img_features, question_features, answer_features), dim=1)
        output = self.fc(combined_features)
        return output

def train_model(model, ARCHITECTURE, train_loader, criterion, optimiser, num_epochs=10):
    print(f'TRAINING %s model' % (ARCHITECTURE))
    model.train()
    
    # Track the overall loss for each epoch
    for epoch in range(num_epochs):
        running_loss = 0.0
        total_batches = len(train_loader)
        start_time = time.time()  # Record start time for the epoch

        for batch_idx, (images, question_embeddings, answer_embeddings, labels) in enumerate(train_loader):
            # Move images/text/labels to the GPU (if available)
            images = images.to(device)          
            question_embeddings = question_embeddings.to(device) 
            answer_embeddings = answer_embeddings.to(device)  
            labels = labels.to(device)

            # Forward pass -- given input data to the model
            outputs = model(images, question_embeddings, answer_embeddings)

            # Calculate loss (error)
            loss = criterion(outputs, labels)  # output should be raw logits
			
            # Backward pass -- given loss above
            optimiser.zero_grad() # clear the gradients
            loss.backward() # computes gradient of the loss/error
            optimiser.step() # updates parameters using gradients
            running_loss += loss.item()

            # Print progress every X batches
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx}/{total_batches}], Loss: {loss.item():.4f}')
        
        # Calculate epoch training time
        epoch_time = time.time() - start_time  # Time taken for this epoch

        # Print average loss and training time for the epoch
        avg_loss = running_loss / total_batches
        print(f'Epoch [{epoch + 1}/{num_epochs}] Average Loss: {avg_loss:.4f}, Time: {epoch_time:.2f} seconds')

    # Calculate total training time
    total_training_time = time.time() - start_time  # Total time for all epochs
    print(f'Total Training Time: {total_training_time:.2f} seconds')

def evaluate_model(model, ARCHITECTURE, test_loader, device):
    print(f'EVALUATING %s model' % (ARCHITECTURE))
    model.eval()
    total_test_loss = 0
    all_labels = []
    all_predictions = []
    reciprocal_ranks = []  # To store reciprocal ranks for MRR calculation
    start_time = time.time()

    with torch.no_grad():
        for images, question_embeddings, answer_embeddings, labels in test_loader:
            # Move images/text/labels to the GPU (if available)
            images = images.to(device)          
            question_embeddings = question_embeddings.to(device) 
            answer_embeddings = answer_embeddings.to(device)  
            labels = labels.to(device)  # Labels are single integers (0 or 1)
			
            # Perform forward pass on our data
            outputs = model(images, question_embeddings, answer_embeddings)
			
            # Accumulate loss on test data
            total_test_loss += criterion(outputs, labels)  

            # Since outputs are logits, apply softmax to get probabilities
            predicted_probabilities = torch.softmax(outputs, dim=1)  # Use softmax for multi-class output
            predicted_class = predicted_probabilities.argmax(dim=1)  # Get the predicted class index (0 or 1)

            # Store labels and predictions for later analysis
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted_class.cpu().numpy())

            # Calculate MRR
            for i in range(len(labels)):
                # Get the predicted probabilities for the current question
                question_probs = predicted_probabilities[i].cpu().numpy()

                # Sort the candidate answers by their predicted probabilities (descending order)
                ranked_indices = np.argsort(question_probs)[::-1]

                # Find the rank of the correct answer
                correct_index = labels[i].item()
                rank = np.where(ranked_indices == correct_index)[0][0] + 1  # Add 1 because ranks start at 1

                # Calculate the reciprocal rank
                reciprocal_rank = 1.0 / rank
                reciprocal_ranks.append(reciprocal_rank)

    # Convert to numpy arrays for easier calculations
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    # Calculate true positives, true negatives, false positives, false negatives
    tp = np.sum((all_predictions == 1) & (all_labels == 1))  # True positives
    tn = np.sum((all_predictions == 0) & (all_labels == 0))  # True negatives
    fp = np.sum((all_predictions == 1) & (all_labels == 0))  # False positives
    fn = np.sum((all_predictions == 0) & (all_labels == 1))  # False negatives

    # Calculate sensitivity, specificity, and balanced accuracy
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    balanced_accuracy = (sensitivity + specificity) / 2.0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)

    # Calculate Mean Reciprocal Rank (MRR)
    mrr = np.mean(reciprocal_ranks)

    elapsed_time = time.time() - start_time
    print(f'Balanced Accuracy: {balanced_accuracy:.4f}, {elapsed_time:.2f} seconds')
    print(f'F1-Score: {f1_score}')
    print(f'Total Test Loss: {total_test_loss:.4f}')
    print(f'Mean Reciprocal Rank (MRR): {mrr:.4f}')

# Main Execution
if __name__ == '__main__':
    # Check GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Paths and files
    IMAGES_PATH = "ITM_Classifier-baselines/visual7w-images"
    train_data_file = "ITM_Classifier-baselines/visual7w-text/v7w.TrainImages.itm.txt"
    dev_data_file = "ITM_Classifier-baselines/visual7w-text/v7w.DevImages.itm.txt"
    test_data_file = "ITM_Classifier-baselines/visual7w-text/v7w.TestImages.itm.txt"
    sentence_embeddings_file = "ITM_Classifier-baselines/v7w.sentence_embeddings-gtr-t5-large.pkl"
    sentence_embeddings = load_sentence_embeddings(sentence_embeddings_file)

    # Create datasets and loaders
    train_dataset = ITM_Dataset(IMAGES_PATH, train_data_file, sentence_embeddings, data_split="train", train_ratio=0.2)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_dataset = ITM_Dataset(IMAGES_PATH, test_data_file, sentence_embeddings, data_split="test")  # whole test data
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # The dev set is not used in this program and you should/could use it for example to optimise your hyperparameters
    #dev_dataset = ITM_Dataset(images_path, "dev_data.txt", sentence_embeddings, data_split="dev")  # whole dev data

    # Create the model using one of the supported architectures
    MODEL_ARCHITECTURE = "CNN"  # Options: "CNN", "ViT", or "CNN_Attention"
    USE_PRETRAINED_MODEL = True
    model = ITM_Model(num_classes=2, ARCHITECTURE=MODEL_ARCHITECTURE, PRETRAINED=USE_PRETRAINED_MODEL).to(device)
    print("\nModel Architecture:")
    print(model)

    # Print the parameters of the model selected above
    total_params = 0
    print("\nModel Trainable Parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:  # print trainable parameters
            num_params = param.numel()  
            total_params += num_params
            print(f"{name}: {param.data.shape} | Number of parameters: {num_params}")
    print(f"\nTotal number of parameters in the model: {total_params}")
    print(f"\nUSE_PRETRAINED_MODEL={USE_PRETRAINED_MODEL}\n")

    # Define loss function and optimiser 
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)

    # Train and evaluate the model
    train_model(model, MODEL_ARCHITECTURE, train_loader, criterion, optimiser, num_epochs=10)
    evaluate_model(model, MODEL_ARCHITECTURE, test_loader, device)
