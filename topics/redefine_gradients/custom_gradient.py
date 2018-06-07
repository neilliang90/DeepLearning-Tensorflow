# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 20:13:06 2018

@author: gul15103
"""
import tensorflow as tf
import tensorflow.contrib.eager as tfe
#tfe.enable_eager_execution()


# Adding a calculation in forward direction but ignore it in backward direction 
#tf.stop_gradient()
t = g(x)
y = t + tf.stop_gradient(f(x) - t)


# Ignore an op in forward direction, but calculate in backward direction
# Example: clipping or scaling gradients

#clipping https://stackoverflow.com/questions/43839431/tensorflow-how-to-replace-or-modify-gradient
input = tfe.Variable([-3.0, 2.0, 0.02], dtype=tf.float32)

@tf.custom_gradient
def clip_grad_layer(x):
  #squared = tf.pow( x, 2)
  def grad(dy):
    return tf.clip_by_value(dy, -0.1, 0.1)
  return tf.identity( x ), grad

with tf.GradientTape( ) as tape:
  output = clip_grad_layer( input )

grad_clip = tape.gradient(output,[input])
print( output)    
print( grad_clip )
