import torch
from time import time
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchinfo import summary

class finalAcc:
    def __init__(self, model):
        # Define transformation for test dataset
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4913996458053589, 0.48215845227241516, 0.44653093814849854),
                                 (0.2470322549343109, 0.24348513782024384, 0.26158788800239563))
        ])
        self.model = model  # Initialize model
        # Check for available device (CPU or GPU)
        if torch.has_mps and torch.backends.mps.is_built() and torch.backends.mps.is_available():  # Check for MacBook support
            self.device = "mps:0"  # Set device as MPS (Mac Pro support)
            print("Device set as MPS")
        elif torch.has_cuda and torch.cuda.is_available():
            self.device = "cuda:0"  # Set device as CUDA (GPU)
            print("Device set as CUDA")
        else:
            self.device = "cpu"  # Set device as CPU if no GPU is available
            print("No GPU available using CPU")
        self.criterion = nn.CrossEntropyLoss()  # Initialize cross-entropy loss

    def getDataLoaders(self):
        # Load CIFAR-10 test dataset
        testset = datasets.CIFAR10(
            root='./data', train=False, download=True, transform=self.transform)
        # Create DataLoader for test dataset
        dataLoader = DataLoader(testset, batch_size=100, shuffle=True, num_workers=0)
        return dataLoader

    def test(self, dataLoader):
        self.model.eval()  # Set model to evaluation mode
        self.model.to(self.device)  # Move model to the specified device
        test_loss = 0
        correct = 0
        total = 0
        print("Starting final Test on all images in CIFAR-10 Test set")
        startTime = time()
        with torch.no_grad():  # Disable gradient calculation
            for batch_idx, (inputs, targets) in enumerate(dataLoader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)  # Compute loss

                test_loss += loss.item()
                _, predicted = outputs.max(1)  # Get predicted labels
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()  # Count correct predictions
        endTime = time()
        # Print test accuracy and testing time
        print("Test Accuracy: " + str(100. * correct / total) + "% | Testing Time: " + str(endTime - startTime) + " sec")

    def finalTest(self):
        dataLoader = self.getDataLoaders()  # Get DataLoader for test dataset
        self.test(dataLoader)  # Perform final testing

if __name__ == "__main__":
    model = torch.jit.load('ResNet.pt')  # Load the pre-trained ResNet model
    summary(model)  # Print model summary
    acc = finalAcc(model)  # Initialize finalAcc object with the model
    acc.finalTest()  # Perform final testing
