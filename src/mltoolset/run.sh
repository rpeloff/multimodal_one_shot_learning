import mltoolset as ml
import tensorflow as tf
import numpy as np
n_data = 5
n_inputs = 28*28
data = np.asarray(np.random.rand(n_data, n_inputs), dtype=ml.NP_FLOAT)
x_input = tf.placeholder(ml.TF_FLOAT, [None, n_inputs])
input_shape = [28, 28, 1]
n_layer_filters = [32, 32, 64, 64, 128]
filter_sizes = [
            # 1st cnn stack (can specify as tuple) 
            (3, 3),  # filter shape of first layer:  32 filters with shape 3x3x1  =   288 weights
            (3, 3),  # filter shape of second layer: 32 filters with shape 3x3x32 = 9,216 weights
            # 2nd cnn stack (can specify as integers) 
            3,       # filter shape of third layer:  64 filters with shape 3x3x32 = 18,432 weights
            3,       # filter shape of fourth layer: 64 filters with shape 3x3x64 = 36,864 weights
            # 3rd cnn stack (can specify as list) 
            [3, 3]   # filter shape of fifth layer:  128 filters with shape 3x3x64 = 73,728 weights
        ]
conv_paddings = 'same'  # Output width/height same as input
pool_sizes = [
            # 1st cnn stack
            None,    # skip pooling in first layer
            (2, 2),  # pool shape of second layer: (2x2)
            # 2nd cnn stack (POOL layer at the end)
            None,    # skip pooling in third layer
            2,       # pool shape of fourth layer: (2x2)
            # 3rd cnn stack (no pooling)
            None     # skip pooling in fifth layer
        ]
activation = tf.nn.relu
keep_prob = 0.5
drop_channels = True
batch_norm = 'before'
training = True
debug = True
x_cnn = ml.neural_blocks.convolutional_layers(
                x_input=x_input,
                input_shape=input_shape,
                n_layer_filters=n_layer_filters,
                layer_kernel_sizes=filter_sizes,
                layer_conv_paddings=conv_paddings,
                layer_pool_sizes=pool_sizes,
                activation=activation,
                keep_prob=keep_prob,
                drop_channels=drop_channels,
                batch_norm=batch_norm,
                training=training,
                debug=debug     
            )
