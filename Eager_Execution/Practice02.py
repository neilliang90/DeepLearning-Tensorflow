# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 19:03:43 2018

@author: gul15103
"""

import tensorflow as tf
import tensorflow.contrib.eager as tfe
#tfe.enable_eager_execution()


@tfe.custom_gradient
def log1pexp(x):
 e = tf.exp(x)
 def grad(dy):
   return dy * (1 - 1 / (1 + e))
 return tf.log(1 + e), grad
grad_log1pexp = tfe.gradients_function(log1pexp)
# Gradient at x = 0 works as before.
#print(grad_log1pexp(0.))
# [0.5]
# And now gradient computation at x=100 works as well.
print(grad_log1pexp(100.))
# [1.0]