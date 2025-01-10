# Learning_PyTorch

A repository where I experiment with PyTorch features.

Contents:
- `cifar_cnn.py` where I follow PyTorch's [tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) to train a simple Convolutional Neural Network to classify images from the CIFAR-10 dataset. 
- `learning_autograd.py` where I experimented with PyTorch's autograd bsaed on another one of their guides, "The Fundamentals of Autograd." It allowed me to see how the computational tree is set up during neural net training. 
- `resNet.py` where I learned to train a [Residual] Network (ResNet) to classify images from the CIFAR-100 datset. The ResNet architecture uses feedforward connections between layers to retain information, especially learned features. This makes learning faster and more reliable, and also allows for much deeper networks without the shortfalls of regular deep neural nets.
    - Like the one in this repository, ResNets contain many convolutional layers that extract various features of the images. 
    - From my own experiments, the ResNet took much longer to train on my CPU than a GPU. So I just simply went to Google Colab, pasted my code, and it ran much faster.