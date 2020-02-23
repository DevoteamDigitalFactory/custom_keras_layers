''' 
tensorflow.__version__ > 2.0
'''
import tensorflow as tf
from tensorflow.python.framework import tensor_shape

class ConcatenateWithCropping2D(tf.keras.layers.Layer):
  def __init__(self, data_format='channels_last', **kwargs):
    super(ConcatenateWithCropping2D, self).__init__(**kwargs)
    self.data_format = data_format
  
  def get_config(self):
    config = super().get_config().copy()
    config.update({
            'data_format': self.data_format
        })
    return config
  def call(self, inputs):
    assert isinstance(inputs, list)
    if self.data_format == 'channels_first':
      try:
        rows = min([x.shape[2] for x in inputs])
      except:
        rows = None
      try:
        cols = min([x.shape[3] for x in inputs])   
      except:
        cols = None   
      return tf.keras.backend.concatenate([
        x[:, :, 0:rows, 0:cols] for x in inputs                                  
      ], axis=1)
    else:
      try:
        rows = min([x.shape[1] for x in inputs])
      except:
        rows = None
      try:
        cols = min([x.shape[2] for x in inputs])  
      except:
        cols = None 
      return tf.keras.backend.concatenate([
        x[:, 0:rows, 0:cols, :] for x in inputs                                  
      ], axis=3)                                      
      

  def build(self, input_shape):
    assert isinstance(input_shape, list)
    if self.data_format == 'channels_first':
      idx_r = 2
      idx_c = 3
      idx_ch = 1
    else:
      idx_r = 1
      idx_c = 2
      idx_ch = 3
    try:
      self.rows = min([shape[idx_r] for shape in input_shape])
    except:
      self.rows = None
    try:
      self.cols = min([shape[idx_c] for shape in input_shape])
    except:
      self.cols = None
    try:
      self.channels = sum([shape[idx_ch] for shape in input_shape])
    except:
      self.channels = None
    super(ConcatenateWithCropping2D, self).build(input_shape)  

  def compute_output_shape(self, input_shape): 
    assert isinstance(input_shape, list)
    if self.data_format == 'channels_first':  
      return tensor_shape.TensorShape([input_shape[0][0], self.channels, self.rows, self.cols])
    else:
      return tensor_shape.TensorShape([input_shape[0][0], self.rows, self.cols, self.channels])
