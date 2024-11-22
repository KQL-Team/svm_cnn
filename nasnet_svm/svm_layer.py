from tensorflow.keras.layers import Layer
class svm_layer(Layer):
    def __init__(self, penalty_para=1.0):
        super(svm_layer, self).__init__()
        self.penalty_para = penalty_para
    def build(self, input_shape):
        self.readout_weight = self.add_weight(name='readout_weight',
                                              shape=(input_shape[-1],),
                                              initializer='normal_normal',
                                              trainable=True)
        self.bias = self.add_weight(name='bias', shape=(1,), initializer='zeros', trainable=True )
    def call(self, inputs):
        output = tf.matmul(inputs, self.readout_weight) + self.bias
        return output
    def hinge_loss(self, y_true, y_pred):
        loss = tf.reduce_mean(tf.maximum(0., 1 - y_true * y_pred))
        regularization_loss = tf.reduce_mean(tf.square(self.readout_weight))
        return self.penalty_para*loss + regularization_loss
