import tensorflow as tf
import onnx
import keras2onnx

model = tf.keras.models.load_model('Model.h5')
onnx_model = keras2onnx.convert_keras(model, model.name)

file = open("model.onnx", "wb")
file.write(onnx_model.SerializeToString())
file.close()
