from sys import exit
from ResNetModel import *  # Importing ResNetModel module
from torchinfo import summary  # Importing summary function from torchinfo module
from TrainAndTest import *  # Importing TrainAndTest module
from torch import optim  # Importing optim module from torch
from PlotsAndGraphs import *  # Importing PlotsAndGraphs module

def main():
    # Initializing PlotsAndGraphs object for visualization
    plots = PlotsAndGraphs()
    
    # Creating the ResNet model with specified configurations
    print("Creating model...")
    """
    Args:
        1) BasicBlock/BottleNeck -> list<string> : Type of residual block
        2) Number of residual layers and blocks -> list<int> : Number of layers and blocks in each layer
        3) Number of channels for each residual block in residual layer -> list<int> : Number of channels for each block
        4) Conv kernel size -> int **DO NOT CHANGE : Kernel size for convolutional layers
        5) Skip connection kernel size -> int : Kernel size for skip connections
        6) Kernel size of Average pooling layer -> int : Kernel size for average pooling layer
    """
    model = ResNet([BasicBlock, BasicBlock, BasicBlock, BasicBlock], [2, 2, 2, 2], [64, 128, 232, 268], 3, 1, 4)  # Reaches ~80% within 5 epochs

    # Checking if the total number of parameters in the model exceeds 5 million
    if summary(model).total_params > 5e+06:
        exit("Total Number of Parameters greater than 5 Million")  # Exiting if the condition is met

    print("Model created")

    # Checking for available GPU and setting the device accordingly
    print("Checking for GPU...")
    if torch.has_mps and torch.backends.mps.is_built() and torch.backends.mps.is_available():  # Checking for MacBook Pro support
        device = "mps:0"  # Setting device as MPS (Mac Pro support)
        print("Device set as MPS")
    elif torch.has_cuda and torch.cuda.is_available():
        device = "cuda:0"  # Setting device as CUDA (GPU)
        print("Device set as CUDA")
    else:
        device = "cpu"  # Setting device as CPU if no GPU is available
        print("No GPU available using CPU")

    # Loading datasets and creating dataloaders with 1 worker for loading data
    print("Loading Datasets and creating Dataloader with 1 worker(s)...")
    datasets = TrainAndTest(plots)
    trainloader, testloader = datasets.getDataLoaders(batch_size=128)
    print("Dataloader ready")

    # Loading loss function, optimizer, and scheduler
    print("Loading Loss, Optimizer and Scheduler...")
    criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss for multiclass classification
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-04)  # SGD optimizer with momentum and weight decay
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)  # CosineAnnealingLR scheduler

    # Training, testing, and saving the model
    print("Training and Saving model...")
    datasets.trainTestAndSave(200, model, device, criterion, optimizer, scheduler, trainloader, testloader)


if __name__ == "__main__":
    main()
