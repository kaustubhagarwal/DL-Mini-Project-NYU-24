# Accuracy Maximization: CIFAR-10 Challenge for ResNet

The ResNet model has facilitated the effective training of deeper neural networks. This project explains how we improved accuracy using the existing ResNet framework and the techniques employed to fine-tune the ResNet model while limiting it to 5 million trainable parameters on the CIFAR-10 dataset. The CIFAR-10 dataset comprises 60,000 color images, each 32x32 pixels, distributed across 10 distinct classes. Additionally, the hyperparameters that most significantly impact the model's performance are explored. **Final model's pt file is ResNet.pt**

![Alt text](https://github.com/navoday01/ResNet5M-CIFAR10/blob/main/assets/CIFAR10-2.png)

## ⚙️ Setup

1. Install all the requirements (PyTorch v1.13.0)
```shell
pip3 install torch torchvision torchaudio torchinfo tensorboard
```
2. Clone the GitHub repository
```shell
git clone https://github.com/kaustubhagarwal/DL-Mini-Project-NYU-24
```
3. Change directory into folder
```shell
cd models
```

## ⏳ Training
Run train script `ResNetTrain.py` to recreate similar model
```shell
cd models
python3 ResNetTrain.py
```
## 🖼 Testing

 To Reproduce the accuracy of the model, run `FinalTesting.py` and **ensure the model is on the right folder and change the path in the file.** This script will normalise the images to right value.
```shell
cd models
python3 FinalTesting.py
```


## 📊 Results
| Sr. No.|    Model Name    |  # Conv Channels at layer  |  Optimizer  |  Params  |  Test Acc  |  File Link  |
|--------|------------------|----------------------------|------------ |----------|-------------|-------------|
|   1    |  Resnet20SGD     |     [64,128,192,256]       |    SGD+M    |   4.47M  |  93.32%     | [LINK](https://github.com/navoday01/ResNet5M-CIFAR10/blob/main/models/weights/Resnet20SGD.pt)            |
|   2    |  Resnet20AdaGrad |     [64,128,192,256]       |    Adagrad  |   4.47M  |  90.55%     | -            |
|   3    |  Resnet20RMSprop |     [64,128,192,256]       |    RMSProp  |   4.47M  |  89.13%     | [LINK](https://github.com/navoday01/ResNet5M-CIFAR10/blob/main/models/weights/ResnetRMSProp.pt)                    |
|   4    |  Resnet20Adam    |     [64,128,192,256]       |    Adam     |   4.47M  |  93.05%     |  -          |
|   5    |  Resnet18Adam    |     [64, 128, 232, 268]    |    Adam     |   4.99M  |  81.03%     |  [LINK](https://github.com/navoday01/ResNet5M-CIFAR10/blob/main/models/weights/Resnet18Adam.pt)                   |
|   **6**    |  **Resnet18SGD**     |     **[64, 128, 232, 268]**    |    **SGD+M**   |   **4.99M**  |  **95.55%**     | [**LINK**](https://github.com/navoday01/ResNet5M-CIFAR10/blob/main/models/weights/Resnet18SGD.pt)                     |
|   7    |  Resnet18Bn      |     [64, 118, 178, 256]    |    SGD+M    |   4.99M  |  91.97%     |  [LINK](https://github.com/navoday01/ResNet5M-CIFAR10/blob/main/models/weights/ResnetBn.pt)                   |
|   8    |  Resnet18BnBb    |     [64, 128, 232, 256]    |    SGD+M    |   4.99M  |  92.39%     |   [LINK](https://github.com/navoday01/ResNet5M-CIFAR10/blob/main/models/weights/ResnetBnBb.pt)                  |
|   9    |  Resnet18BbBn    |     [64, 100, 160, 256]    |    SGD+M    |   4.99M  |  92.73%     |      -       |
|   10   |  Resnet56SGD     |     [64, 96, 128, 190]     |    SGD+M    |   4.98M  |  95.51%     |  [LINK](https://github.com/navoday01/ResNet5M-CIFAR10/blob/main/models/weights/Resnet56SGD.pt)                   |
|   11   |  Resnet56Adam    |     [64, 128, 232, 268]    |    Adam     |   4.98M  |  93.37%     | [LINK](https://github.com/navoday01/ResNet5M-CIFAR10/blob/main/models/weights/Resnet56Adam.pt)                    |
|   12   |  Resnet156       |     [64, 72, 74, 90]       |    SGD+M    |   4.99M  |  93.82%     |  [LINK](https://github.com/navoday01/ResNet5M-CIFAR10/blob/main/models/weights/Resnet156.pt)                   |

## 📑 Report
To read a detailed report, click [HERE](assets/Report.pdf)

## 📦 Conclusion

The final model, constrained to under 5 million parameters and utilizing SGD (Stochastic Gradient Descent), reached an accuracy of 95.55% on the CIFAR-10 test dataset. This achievement was accomplished by systematically tuning the hyperparameters and optimizers.

## 👩‍⚖️ Acknowledgement

We thank professor Chinmay Hegde for his guidance, mentorship, and expertise throughout this project. We would also like to thank the staff and facilities at New York University for providing us with the necessary resources to see this project to completion. 



