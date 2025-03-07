import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    # Check if GPU is available (cuda for gpu, mps for mac m1, cpu for others)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print("Using {} device".format(device))

    # Hyperparameters
    learning_rate = 0.0001 
    epochs = 100
    batch_size = 128
    num_classes = 10  # 10 classes for CIFAR10

    # Define the transform function for trainset
    transform_train = transforms.Compose(
        [
            # data augmentation
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            # data normalization
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # Define the transform function for val
    transform_val = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # Load the CIFAR10 dataset
    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    valset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_val)

    # Define the dataloader
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Load pretrained VGG19 model with batch normalization
    model = torchvision.models.vgg19_bn(pretrained=True)
    # Change the last layer to fit the CIFAR10 dataset
    model.classifier[6] = nn.Linear(4096, num_classes)

    # Move the model to GPU for calculation
    model = model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)

    # Record the loss and accuracy for training and validation set
    train_loss, train_acc = [], []
    val_loss, val_acc = [], []

    
    # Train the model
    print("Training the model...")
    for epoch in tqdm(range(epochs)):
        model.train()
        print("\nEpoch: ", epoch + 1)
        batch_train_loss, batch_train_acc = [], []
        for i, data in enumerate(train_loader):
            # Prepare data
            inputs, labels = data
            # Load data to GPU
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Record the loss and accuracy for every batch
            batch_train_loss.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            batch_train_acc.append((predicted == labels).sum().item() / batch_size)

        # Record the loss and accuracy for every epoch
        loss = np.mean(batch_train_loss)
        acc = np.mean(batch_train_acc)
        train_loss.append(loss)
        train_acc.append(acc)
        print(f"Training Loss: {loss} | Training Accuracy: {acc}")

        # Validate the model
        model.eval()
        with torch.no_grad():
            batch_val_loss, batch_val_acc = [], []
            correct = [0] * num_classes
            total = [0] * num_classes

            for i, data in enumerate(val_loader):
                # Prepare data
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                # Record the loss and accuracy for every batch
                batch_val_loss.append(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                batch_val_acc.append((predicted == labels).sum().item() / batch_size)

                # Per-class accuracy calculation
                for label, prediction in zip(labels, predicted):
                    if label == prediction:
                        correct[label] += 1
                    total[label] += 1

            # Record the loss and accuracy for every epoch
            loss = np.mean(batch_val_loss)
            acc = np.mean(batch_val_acc)
            val_loss.append(loss)
            val_acc.append(acc)
            print(f"Validation Loss: {loss} | Validation Accuracy: {acc}")

            # Print accuracy per class
            classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
            for i in range(num_classes):
                if total[i] > 0:  # Avoid division by zero
                    print(f"Accuracy for class {classes[i]}: {100 * correct[i] / total[i]:.2f}%")
                else:
                    print(f"Accuracy for class {classes[i]}: N/A (no samples)")
        
        val_loss_epoch = np.mean(batch_val_loss)

        # Step the scheduler
        scheduler.step(val_loss_epoch)
        
        # Log the learning rate
        for param_group in optimizer.param_groups:
            print(f"Current Learning Rate: {param_group['lr']}")
            
        # Checkpoint Saving
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'./model_epoch_{epoch + 1}.pth')

    # Plot the loss and accuracy
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label="train_loss")
    plt.plot(val_loss, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_acc, label="train_acc")
    plt.plot(val_acc, label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend()

    # Save the image
    plt.savefig(f"./epoch_{epochs}.png")
    # Show the image
    plt.show()

    # Save the model
    torch.save(model.state_dict(), f"./model.pth")
