import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from net import Model

def train_epoch(model: Model, loader: DataLoader, criterion, optimizer, device):
    """Train for one epoch and return average loss and accuracy"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for x, y in tqdm(loader, desc="Training", leave=False):
        x = x.transpose(1, 2).to(device)  # (B, 3, 132)
        y = y.to(device)
        
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * x.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(y).sum().item()
        total += y.size(0)
    
    avg_loss = total_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def validate(model, loader, criterion, device):
    """Validate and return average loss and accuracy"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Validation", leave=False):
            x = x.transpose(1, 2).to(device)  # (B, 3, 132)
            y = y.to(device)
            
            outputs = model(x)
            loss = criterion(outputs, y)
            
            total_loss += loss.item() * x.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(y).sum().item()
            total += y.size(0)
    
    avg_loss = total_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def train_model(model, train_loader, val_loader, num_epochs=50, lr=0.001, device='cuda'):
    """Main training function with validation"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5 
    )
    
    best_val_acc = 0
    best_model_state = None
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    print(f"Starting training on {device}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"{'='*60}\n")
    
    for epoch in range(num_epochs):
        # Training
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  Best Val Acc: {best_val_acc:.2f}%\n")
    
    # Load best model
    model.load_state_dict(best_model_state)
    print(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%")
    
    return model, history
