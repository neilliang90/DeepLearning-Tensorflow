#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 20:06:50 2018

@author: Neil
"""

#Import data into tensorflow: https://www.tensorflow.org/get_started/datasets_quickstart

# 1, Reading in-memory data from numpy arrays.

import tensorflow as tf
def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset

import iris_data

# Fetch the data
train, test = iris_data.load_data()
features, labels = train


batch_size=100
iris_data.train_input_fn(features, labels, batch_size)