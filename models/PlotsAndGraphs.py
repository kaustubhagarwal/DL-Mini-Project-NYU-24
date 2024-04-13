import torchvision
from torch.utils.tensorboard import SummaryWriter

class PlotsAndGraphs:
    def __init__(self):
        # Initialize SummaryWriter for TensorBoard visualization
        self.writer = SummaryWriter('runs/ResNet')

    def plotRandomImg(self, dataLoader):
        # Get a batch of images from the DataLoader
        dataiter = iter(dataLoader)
        images, labels = next(dataiter)
        # Create a grid of images
        img_grid = torchvision.utils.make_grid(images)
        # Add the grid of images to TensorBoard
        self.writer.add_image("Images of CIFAR-10", img_grid)

    def graph(self, model, image):
        # Add the computational graph of the model to TensorBoard
        self.writer.add_graph(model, image)

    def plot(self, name, value, epoch):
        # Add scalar value (e.g., loss or accuracy) to TensorBoard for visualization
        self.writer.add_scalar(str(name), value, epoch)
