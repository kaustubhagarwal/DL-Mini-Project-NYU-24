import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, conv_kernel_size=3, skip_connection_kernel_size=1):
        super(BasicBlock, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=conv_kernel_size, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=conv_kernel_size, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection to adjust dimensions
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=skip_connection_kernel_size, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # First convolutional layer with batch normalization and ReLU activation
        out = self.bn2(self.conv2(out))  # Second convolutional layer with batch normalization
        out += self.shortcut(x)  # Shortcut connection
        out = F.relu(out)  # ReLU activation
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, conv_kernel_size=3, skip_connection_kernel_size=1):
        super(Bottleneck, self).__init__()
        # First 1x1 convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # Second 3x3 convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=conv_kernel_size, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # Third 1x1 convolutional layer
        self.conv3 = nn.Conv2d(out_channels, self.expansion * out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * out_channels)

        # Shortcut connection to adjust dimensions
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=skip_connection_kernel_size, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # First 1x1 convolutional layer with batch normalization and ReLU activation
        out = F.relu(self.bn2(self.conv2(out)))  # Second 3x3 convolutional layer with batch normalization and ReLU activation
        out = self.bn3(self.conv3(out))  # Third 1x1 convolutional layer with batch normalization
        out += self.shortcut(x)  # Shortcut connection
        out = F.relu(out)  # ReLU activation
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, channels, conv_kernel_size, skip_connection_kernel_size, avg_maxPool_size, num_classes=10):
        super(ResNet, self).__init__()
        self.input = channels[0]
        self.conv_kernel_size = conv_kernel_size
        self.skip_connection_kernel_size = skip_connection_kernel_size
        self.avg_maxPool_size = avg_maxPool_size

        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])

        # Residual layers
        self.layer1 = self._make_layer(block[0], channels[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block[1], channels[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block[2], channels[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block[3], channels[3], num_blocks[3], stride=2)

        # Output layer
        self.linear = nn.Linear(channels[3] * block[3].expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.input, out_channels, stride, self.conv_kernel_size, self.skip_connection_kernel_size))
            self.input = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # Initial convolutional layer with batch normalization and ReLU activation
        # Pass through residual layers
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        # Global average pooling
        out = F.avg_pool2d(out, self.avg_maxPool_size)
        out = torch.flatten(out, 1)
        out = self.linear(out)  # Output layer
        return out
