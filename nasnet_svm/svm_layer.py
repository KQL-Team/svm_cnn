from tensorflow.keras.layers import Layer
class SVM(Layer):
    def __init__(self, penalty_para=1.0):
        super(svm_layer, self).__init__()
        self.penalty_para = penalty_para
    def build(self, input_shape):
        self.readout_weight = self.add_weight(name='readout_weight',
                                              shape=(input_shape[-1], 1),
                                              initializer='normal_normal',
                                              trainable=True)
        self.bias = self.add_weight(name='bias', shape=(1,), initializer='zeros', trainable=True )
    def call(self, inputs):
        output = tf.matmul(inputs, self.readout_weight) + self.bias
        return output



class MultiClassSVM(Layer):
    def __init__(self, num_classes, penalty_para=1.0):
        super(MultiClassSVM, self).__init__()
        self.num_classes = num_classes
        self.penalty_para = penalty_para
        self.svm_layers = []

    def build(self, input_shape):
        for _ in range(self.num_classes):
            self.svm_layers.append(SVM(penalty_para=self.penalty_para))
        super().build(input_shape)

    def call(self, inputs):
        outputs = []
        for svm in self.svm_layers:
            outputs.append(svm(inputs))
        return tf.stack(outputs, axis=-1)
