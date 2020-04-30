import tensorflow as tf
import hyperparameters as hp
from tensorflow.keras.layers import \
        Conv2D, MaxPool2D, Dropout, Flatten, Dense

class YourModel(tf.keras.Model):
    """ Your own neural network model. """

    def __init__(self):
        super(YourModel, self).__init__()

        # Optimizer
        self.optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=hp.learning_rate,
            momentum=hp.momentum)

        
        self.architecture = [
        Conv2D(64, 3, 1, padding="same", activation="relu", name="block1_conv1"),
        MaxPool2D(2, name="block1_pool"),
        Conv2D(128, 3, 1, padding="same", activation="relu", name="block2_conv1"),
        MaxPool2D(2, name="block2_pool"),
        Conv2D(128, 3, 1, padding="same", activation="relu", name="block3_conv1"),
        MaxPool2D(2, name="block3_pool"),
        Dropout(rate=0.5),
        Flatten(),
        Dense(units=128, activation="relu"),
        #Dropout(rate=0.5),
        Dense(units=15, activation="softmax")
        ]

    def call(self, img):
        """ Passes input image through the network. """

        for layer in self.architecture:
            img = layer(img)

        return img

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for the model. """

        return tf.keras.losses.sparse_categorical_crossentropy(
            labels, predictions, from_logits=False)
