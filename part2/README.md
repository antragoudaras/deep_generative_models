# Adversarial Autoencoders

We will train a Adversarial Autoencoders (`AAE`) model on generating MNIST images. The code is structured in the following way:

* `mnist.py`: Contains a function for preparing the dataset and providing a data loader for training.
* `models.py`: Contains template classes for the Encoder, Decoder, Discriminator and the overall Adversarial Autoencoder.
* `train.py`: Contains the overall procedure of assignment, it parses terminal commands by user, then it sets the hyper-parameters, load dataset, initialize the model, set the optimizers and then
  it trains the adversarial auto-encoder and saves the network generations for each epoch.   
* `unittests.py`: Contains unittests for the Encoder, Decoder, Discriminator networks. It will hopefully help you debugging your code. Your final code should pass these unittests.
* `utils.py`: Contains logging utilities for Tensorboard.

The code is structured in the following way:

* In `models.py`, the Encoder, Decoder, Discriminator and the overall Adversarial Autoencoder network is implemented. Also in Adversarial Autoencoder all required losses are developed in the function `get_loss_discriminator` and `get_loss_autoencoder`.
* In `train.py`, the `main` function defines the required optimizers. In function `train_aae` the discriminator and autoencoder losses are calculated.
  Then one optimization step for both autoencoder and discriminator parts is performed.
  
Default hyper-parameters are provided in the `ArgumentParser` object of the respective training functions. Feel free to play around with those to familiarize yourself with the effect of different hyper-parameters.
The training time with the default hyper-parameters and architecture is less than 30 minutes on a NVIDIA GTX1080Ti.

