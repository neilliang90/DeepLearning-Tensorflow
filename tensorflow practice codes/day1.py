#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 22:50:32 2018

@author: Neil
"""

from tensorflow.python.client import device_lib

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


# Use TensorBoard
import tensorflow as tf
a = tf.constant(2)
b = tf.constant(3)
x = tf.add(a, b)
with tf.Session() as sess:
  #add this line to use TensorBoard.
  writer = tf.summary.FileWriter('./graphs', sess.graph)
  print(sess.run(x))
writer.close() # close the writer when you’re done using it