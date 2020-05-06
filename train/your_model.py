import tensorflow as tf
import hyperparameters as hp
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense

class YourModel(tf.keras.Model):
    """ Your own neural network model. """

    def __init__(self):
        super(YourModel, self).__init__()

        # Optimizer
        self.optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=hp.learning_rate,
            momentum=hp.momentum)

        self.architecture = [
        # Block 1
        Conv2D(16, 3, 1, padding="same", activation="relu", name="block1_conv1"),
        Conv2D(16, 3, 1, padding="same", activation="relu", name="block1_conv2"),
        MaxPool2D(2, name="block1_pool"),
        # Block 2
        Conv2D(32, 3, 1, padding="same", activation="relu", name="block2_conv1"),
        Conv2D(32, 3, 1, padding="same", activation="relu", name="block2_conv2"),
        MaxPool2D(2, name="block2_pool"),
        # Block 3
        Conv2D(64, 3, 1, padding="same", activation="relu", name="block3_conv1"),
        Conv2D(64, 3, 1, padding="same", activation="relu", name="block3_conv2"),
        MaxPool2D(2, name="block3_pool"),
        Dropout(0.5),
        Flatten(),
        Dense(13,activation='softmax')
        ]
        
        # self.architecture = [
        # # Block 1
        # Conv2D(64, 3, 1, padding="same", activation="relu", name="block1_conv1"),
        # Conv2D(64, 3, 1, padding="same", activation="relu", name="block1_conv2"),
        # MaxPool2D(2, name="block1_pool"),
        # # Block 2
        # Conv2D(128, 3, 1, padding="same", activation="relu", name="block2_conv1"),
        # Conv2D(128, 3, 1, padding="same", activation="relu", name="block2_conv2"),
        # MaxPool2D(2, name="block2_pool"),
        # # Block 3
        # Conv2D(256, 3, 1, padding="same", activation="relu", name="block3_conv1"),
        # Conv2D(256, 3, 1, padding="same", activation="relu", name="block3_conv2"),
        # Conv2D(256, 3, 1, padding="same", activation="relu", name="block3_conv3"),
        # MaxPool2D(2, name="block3_pool"),
        # # Block 4
        # Conv2D(512, 3, 1, padding="same", activation="relu", name="block4_conv1"),
        # Conv2D(512, 3, 1, padding="same", activation="relu", name="block4_conv2"),
        # Conv2D(512, 3, 1, padding="same", activation="relu", name="block4_conv3"),
        # MaxPool2D(2, name="block4_pool"),
        # Dropout(0.3),
        # Flatten(),
        # Dense(13,activation='softmax')
        # ]

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
