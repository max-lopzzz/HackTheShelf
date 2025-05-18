import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import time
import copy
from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from PIL import Image


def train_model(model, dataloaders, criterion, optimizer, scheduler=None, num_epochs=25, device='cuda'):
    """
    Trains the Swin Transformer model and returns the best model weights.
    
    Args:
        model: Swin Transformer model
        dataloaders: Dictionary with 'train' and 'val' dataloaders
        criterion: Loss function
        optimizer: Optimizer for training
        scheduler: Learning rate scheduler (optional)
        num_epochs: Number of epochs to train
        device: Device to train on ('cuda' or 'cpu')
        
    Returns:
        model: Best model based on validation accuracy
    """
    since = time.time()
    
    # Initialize variables to track best model
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # Track metrics
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Move model to device
    model = model.to(device)
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
                
            running_loss = 0.0
            running_corrects = 0
            dataset_size = 0
            
            # Iterate over data
            pbar = tqdm(dataloaders[phase], desc=f'{phase}')
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                batch_size = inputs.size(0)
                running_loss += loss.item() * batch_size
                running_corrects += torch.sum(preds == labels.data)
                dataset_size += batch_size
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': running_loss / dataset_size,
                    'acc': running_corrects / dataset_size
                })
            
            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects / dataset_size
            
            # Update scheduler if provided
            if phase == 'train' and scheduler is not None:
                scheduler.step()
                
            # Track metrics
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Deep copy the model if best validation accuracy
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # Plot training history
    plot_training_history(history)
    
    return model, history


def plot_training_history(history):
    """Plot training and validation metrics."""
    plt.figure(figsize=(12, 5))
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()


def evaluate_model(model, dataloader, criterion, device='cuda', class_names=None):
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained model
        dataloader: Test dataloader
        criterion: Loss function
        device: Device to evaluate on
        class_names: List of class names for confusion matrix labels
        
    Returns:
        test_loss: Average loss on test data
        test_acc: Accuracy on test data
    """
    model.eval()
    model = model.to(device)
    
    # Initialize variables
    running_loss = 0.0
    running_corrects = 0
    all_preds = []
    all_labels = []
    
    # Evaluate model
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Testing'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            # Store predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    test_loss = running_loss / len(dataloader.dataset)
    test_acc = running_corrects / len(dataloader.dataset)
    
    print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')
    
    # Confusion matrix and classification report
    if class_names is not None:
        cm = confusion_matrix(all_labels, all_preds)
        #report = classification_report(all_labels, all_preds, target_names=class_names)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        # Add text annotations
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                        horizontalalignment="center",
                        color="white" if cm[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig('confusion_matrix.png')
        plt.show()
        
        #print("Classification Report:")
        #print(report)
    
    return test_loss, test_acc


def save_model(model, optimizer, epoch, filename='swin_transformer_checkpoint.pth'):
    """Save model checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(checkpoint, filename)
    print(f"Model saved to {filename}")


def load_model(model, optimizer=None, filename='swin_transformer_checkpoint.pth', device='cuda'):
    """Load model checkpoint."""
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    print(f"Model loaded from {filename} (epoch {epoch})")
    
    return model, optimizer, epoch


def visualize_predictions(model, dataloader, class_names, device='cuda', num_images=8):
    """Visualize model predictions on random images."""
    model.eval()
    model = model.to(device)
    
    # Get a batch of images
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    
    # Make predictions
    with torch.no_grad():
        outputs = model(images.to(device))
        _, preds = torch.max(outputs, 1)
    
    # Plot images with predictions
    fig = plt.figure(figsize=(15, 8))
    for idx in range(min(num_images, len(images))):
        ax = fig.add_subplot(2, num_images//2, idx+1, xticks=[], yticks=[])
        img = images[idx].cpu().numpy().transpose((1, 2, 0))
        
        # Denormalize if needed
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        ax.imshow(img)
        
        # Color based on prediction correctness
        color = 'green' if preds[idx] == labels[idx] else 'red'
        ax.set_title(f'P: {class_names[preds[idx]]}\nT: {class_names[labels[idx]]}', color=color)
    
    plt.tight_layout()
    plt.savefig('prediction_examples.png')
    plt.show()

class YOLOToClassificationDataset(Dataset):
    """
    Converts YOLO format dataset (used for object detection) to a classification dataset.
    
    For YOLO datasets:
    - Images are in 'images' folder
    - Labels are in 'labels' folder with .txt files (class_id x_center y_center width height)
    
    This loader will:
    1. Read images from the images folder
    2. Use the first class_id in each label file as the classification target
    3. Apply appropriate transforms for the Swin Transformer
    """
    def __init__(self, root_dir, transform=None, target_size=(224, 224)):
        """
        Args:
            root_dir: Path to dataset folder (e.g., 'train', 'valid', or 'test')
            transform: Optional transforms to apply
            target_size: Size to resize images to
        """
        self.root_dir = root_dir
        self.transform = transform
        self.target_size = target_size
        
        self.images_dir = os.path.join(root_dir, 'images')
        self.labels_dir = os.path.join(root_dir, 'labels')
        
        # Get all image files
        self.image_files = [f for f in os.listdir(self.images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Get unique class IDs and create a mapping
        self.classes = self._get_unique_classes()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
    def _get_unique_classes(self):
        """Extract all unique class IDs from the label files."""
        unique_classes = set()
        
        for img_file in self.image_files:
            # Get corresponding label file
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(self.labels_dir, label_file)
            
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        # YOLO format: class_id x_center y_center width height
                        parts = line.strip().split()
                        if parts:
                            class_id = int(parts[0])
                            unique_classes.add(class_id)
        
        return sorted(list(unique_classes))
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Get image path
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        # Read image
        image = Image.open(img_path).convert('RGB')
        
        # Get corresponding label file
        label_file = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.labels_dir, label_file)
        
        # Default class (in case label file doesn't exist)
        class_id = 0
        
        # Read class from label file (use first object's class)
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                line = f.readline().strip()
                if line:
                    class_id = int(line.split()[0])
                    
        # Apply transforms
        if self.transform:
            image = self.transform(image)
            
        return image, class_id


def create_dataloaders_from_yolo(data_root, batch_size=32, num_workers=4):
    """
    Create dataloaders from a YOLO-formatted dataset.
    
    Args:
        data_root: Root directory containing 'train', 'valid', and 'test' folders
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        
    Returns:
        dataloaders: Dictionary with 'train', 'val', and 'test' dataloaders
        class_names: List of class names (as integers from YOLO format)
    """
    # Define transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    
    # Create datasets
    image_datasets = {
        'train': YOLOToClassificationDataset(
            os.path.join(data_root, 'train'), 
            transform=data_transforms['train']
        ),
        'val': YOLOToClassificationDataset(
            os.path.join(data_root, 'valid'), 
            transform=data_transforms['val']
        )
    }
    
    # Add test dataset if it exists
    test_dir = os.path.join(data_root, 'test')
    if os.path.exists(test_dir):
        image_datasets['test'] = YOLOToClassificationDataset(
            test_dir,
            transform=data_transforms['test']
        )
    
    # Create dataloaders
    dataloaders = {
        x: DataLoader(
            image_datasets[x], 
            batch_size=batch_size,
            shuffle=(x == 'train'),
            num_workers=num_workers
        )
        for x in image_datasets.keys()
    }
    
    # Get class names from the training dataset
    class_names = image_datasets['train'].classes
    
    return dataloaders, class_names


def main():
    """Example usage with YOLO dataset."""
    from model import SwinTransformer  # Import your Swin Transformer model
    
    # Set device
    device = torch.device('mps')
    print(f"Using {device} device")
    
    # Load datasets
    data_root = '/Users/ricardosalgadob/Desktop/HackTheShelf2025/YOLO'  # Contains train, valid, test folders
    dataloaders, class_names = create_dataloaders_from_yolo(data_root, batch_size=32)
    
    # Initialize model with the correct number of classes
    num_classes = len(class_names)
    print(f"Training on {num_classes} classes: {class_names}")
    
    model = SwinTransformer(
        img_size=224,
        patch_size=4,
        in_channels=3,
        num_classes=num_classes,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.
    )
    
    # Define loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.05)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    
    # Train model
    model, history = train_model(
        model,
        dataloaders={'train': dataloaders['train'], 'val': dataloaders['val']},
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=10,
        device=device
    )
    
    # Save the trained model
    #save_model(model, optimizer, 100, 'swin_transformer_final.pth')
    
    # Evaluate on test set if available
    if 'test' in dataloaders:
        test_loss, test_acc = evaluate_model(
            model,
            dataloaders['test'],
            criterion,
            device=device,
            class_names=[str(c) for c in class_names]
        )
        
        print(f'Test loss: {test_loss} | Test Acc: {test_acc}')


if __name__ == '__main__':
    main()