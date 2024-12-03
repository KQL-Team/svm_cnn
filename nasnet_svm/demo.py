import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
import numpy as np
import tensorflow as tf
predicted_class = ['Organic ', 'Recyclable']
class SVM(tf.keras.layers.Layer):
    def __init__(self, penalty_para=1.0, **kwargs):
        super(SVM, self).__init__(**kwargs)
        self.penalty_para = penalty_para
    def build(self, input_shape):
        self.readout_weight = self.add_weight(name='readout_weight',
                                              shape=(input_shape[-1], 1),
                                              initializer='random_normal',
                                              trainable=True)
        self.bias = self.add_weight(name='bias', shape=(1,), initializer='zeros', trainable=True )
    def call(self, inputs):
        output = tf.matmul(inputs, self.readout_weight) + self.bias
        return output
model = load_model('nasnet_svm.h5', custom_objects={'svm_layer': SVM})
def browse_file():
    filename = filedialog.askopenfilename()
    if filename:
        img = Image.open(filename)
        img = img.convert('RGB').resize((331, 331))
        img_tk = ImageTk.PhotoImage(img)
        label_img.config(image=img_tk)
        label_img.image = img_tk
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array.astype('float32')
        img_array /= 255.0

        prediction = model.predict(img_array)
        prediction = np.where(prediction > 0, 1, 0)
        label_prediction.config(text=f"Predicted Class: {predicted_class[prediction[0][0]]}")
root = tk.Tk()
root.title("File Browser")
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Kích thước cửa sổ
window_width = 512
window_height = 512

position_top = int(screen_height / 2 - window_height / 2)
position_left = int(screen_width / 2 - window_width / 2)

root.geometry(f"{window_width}x{window_height}+{position_left}+{position_top}")
btn_browse = tk.Button(root, text="Browse", command=browse_file)
btn_browse.pack(pady=20)
label_img = tk.Label(root)
label_img.pack(pady=20)
label_prediction = tk.Label(root, text="Predicted Class")
label_prediction.pack(pady=20)

root.mainloop()