import tensorflow as tf
class CNN(tf.keras.Model):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layer_1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=5,
            activation='relu',
            kernel_initializer='he_normal',
        )
        self.pool_layer_1 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)
        self.drop_out_1 = tf.keras.layers.Dropout(0.2)
        self.conv_layer_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=5, activation='relu', kernel_initializer='he_normal')
        self.pool_layer_2 = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)
        self.drop_out_2 = tf.keras.layers.Dropout(0.2)
        self.flatten = tf.keras.layers.Flatten()
        self.fcl_1 = tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_normal')
        self.drop_out_3 = tf.keras.layers.Dropout(0.25)
        self.fcl_2 = tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_normal')
        self.output_layer = tf.keras.layers.Dense(1, activation='sigmoid')
    def call(self, inputs):
        x = self.conv_layer_1(inputs)
        x = self.pool_layer_1(x)
        x = self.drop_out_1(x)

        x = self.conv_layer_2(x)
        x = self.pool_layer_2(x)
        x = self.drop_out_2(x)

        x = self.flatten(x)
        x = self.fcl_1(x)
        x = self.drop_out_3(x)
        return self.output_layer(x)
