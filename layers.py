''' 
tensorflow.__version__ > 2.0
'''
import tensorflow as tf
from tensorflow.python.framework import tensor_shape

class ConcatenateWithCropping2D(tf.keras.layers.Layer):
  '''
  Concatenate a list of feature maps with different shape.
  Output shape is the mininum shape among the list, features map are cropped 
  to this miminum shape before being concatenated
  '''
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
    min_shape = tf.reduce_min([tf.shape(i) for i in inputs], axis=0)
    if self.data_format == 'channels_first':
      min_shape = tf.concat([tf.constant([-1]), tf.constant([-1]), min_shape[2:4]], axis=0)
      return tf.ensure_shape(
          tf.keras.backend.concatenate([
            tf.slice(x, (0,0,0,0), min_shape) for x in inputs                                  
          ], axis=3),
          shape=(None, None, sum([x.shape[1] for x in inputs]), None)
      )  
    else:
      min_shape = tf.concat([tf.constant([-1]),min_shape[1:3], tf.constant([-1])], axis=0)
      return tf.ensure_shape(
          tf.keras.backend.concatenate([
            tf.slice(x, (0,0,0,0), min_shape) for x in inputs                                  
          ], axis=3),
          shape=(None, None, None, sum([x.shape[3] for x in inputs]))
      )                                      
      

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
