# Learning_PyTorch

A repository where I experiment with computer vision algorithms in PyTorch.

Contents:
- `cifar_cnn.py` where I follow PyTorch's [tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html) to train a simple Convolutional Neural Network to classify images from the CIFAR-10 dataset. 
- `learning_autograd.py` where I experimented with PyTorch's autograd bsaed on another one of their guides, "The Fundamentals of Autograd." It allowed me to see how the computational tree is set up during neural net training. 
- `resNet.py` where I learned to train a [Residual](https://arxiv.org/pdf/1512.03385) Network (ResNet) to classify images from the CIFAR-100 datset. The ResNet architecture uses feedforward connections between layers to retain information, especially learned features. This makes learning faster and more reliable, and also allows for much deeper networks without the shortfalls of regular deep neural nets.
    - Like the one in this repository, ResNets contain many convolutional layers that extract various features of the images. 
    - From my own experiments, the ResNet took much longer to train on my CPU than a GPU. So I just simply went to Google Colab, pasted my code, and it ran much faster.
    - At first, the network has only reached an accuracy of 51.9 % after 30 epochs (taking 43 minutes). 
    - By changing the optimizer to Adam and using a base learning rate of 0.01, the *train* accuracy after 30 epochs taking the same time was 96.15%. 
    - A further boost to the accuracy was the careful adjustment of the learning rate to lower it (as Adam likes) to 0.002. That resulted in a final accuracy of about 98.24% on the training dataset.
    - To take it even further, I used a learning rate scheduler to anneal the rate by 0.6 every 10 epochs. That made the final training accuracy 99.7%.
    - There's one caveat...the test dataset accuracy was poor, about 65%. Needs improvement!

Upcoming:
- A Variational Autoencoder