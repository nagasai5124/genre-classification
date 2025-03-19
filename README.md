# genre-classification
# BERT (Bidirectional Encoder Representations from Transformers) 
leverages a transformer-based neural network to understand and generate human-like language. BERT employs an encoder-only architecture. In the original Transformer architecture, there are both encoder and decoder modules. The decision to use an encoder-only architecture in BERT suggests a primary emphasis on understanding input sequences rather than generating output sequences. BERT is designed to generate a language model so, only the encoder mechanism is used. Sequence of tokens are fed to the Transformer encoder. These tokens are first embedded into vectors and then processed in the neural network. The output is a sequence of vectors, each corresponding to an input token, providing contextualized representations.


To reduce the overfitting problem:
# Early Stopping for PyTorch
Early stopping is a form of regularization used to avoid overfitting on the training dataset. Early stopping keeps track of the validation loss, if the loss stops decreasing for several epochs in a row the training stops. The EarlyStopping class in early_stopping_pytorch/early_stopping.py is used to create an object to keep track of the validation loss while training a PyTorch model. It will save a checkpoint of the model each time the validation loss decrease. We set the patience argument in the EarlyStopping class to how many epochs we want to wait after the last time the validation loss improved before breaking the training loop. There is a simple example of how to use the EarlyStopping class in the MNIST_Early_Stopping_example notebook.

Underneath is a plot from the example notebook, which shows the last checkpoint made by the EarlyStopping object, right before the model started to overfit. It had patience set to 20.

# checkpointing
periodically saving a model's state (weights, biases, etc.) during training to allow resuming from a previous point if training is interrupted or needs to be restarted. This ensures that progress isn't lost and training can continue from where it left off. 
