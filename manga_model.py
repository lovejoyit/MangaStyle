import tensorflow as tf


def model(images, batch_size, classes, dropout):
    """Build the model
    
    Args:
        images: Tensor with image batch [batch_size, height, width, channels].
        batch_size: Number of image of one batch.
        classes: Number of classes.
        dropout: Dropout probability, but does not use drop out in this model.
    Returns:
        softmax_linear: Tensor with the computed logits.
    """    
    # Convolution_layer1
    with tf.variable_scope('convolution_layer1') as scope:
        weights = tf.get_variable('weights', 
                                  shape = [3, 3, 3, 32],
                                  dtype = tf.float32, 
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biasrq12ges2', 
                                 shape=[32],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME')
        conv_biases = tf.nn.bias_add(conv, biases)
        conv_layer1 = tf.nn.relu(conv_biases, name = 'conv1')
    
    # Maxpooling1_layer1  
    with tf.variable_scope('maxpooling1_layer1') as scope:
        maxpool1 = tf.nn.max_pool(conv_layer1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='maxpooling1')
        #maxpool1 = tf.nn.dropout(maxpool1, dropout)

    
    # Convolution_layer2
    with tf.variable_scope('convolution_layer2') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3, 3, 32, 64],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[64], 
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(maxpool1, weights, strides=[1, 1, 1, 1], padding='SAME')
        conv_biases = tf.nn.bias_add(conv, biases)
        conv_layer2 = tf.nn.relu(conv_biases, name = 'conv2')
    
    
    # Maxpooling1_layer2
    with tf.variable_scope('maxpooling1_layer2' , reuse=True) as scope:
        maxpool2 = tf.nn.max_pool(conv_layer2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME',name='maxpooling2')
        #maxpool2 = tf.nn.dropout(maxpool2, dropout)
    
    # Convolution_layer3
    with tf.variable_scope('convolution_layer3') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3, 3, 64, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[128], 
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(maxpool2, weights, strides=[1, 1, 1, 1], padding='SAME')
        conv_biases = tf.nn.bias_add(conv, biases)
        conv_layer3 = tf.nn.relu(conv_biases, name = 'conv3')
    
    # Convolution_layer4
    with tf.variable_scope('convolution_layer4') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3, 3, 128, 128],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[128], 
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(conv_layer3, weights, strides=[1,1,1,1],padding='SAME')
        conv_biases = tf.nn.bias_add(conv, biases)
        conv_layer4 = tf.nn.relu(conv_biases, name = 'conv4')
    
    # Convolution_layer5
    with tf.variable_scope('convolution_layer5') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3, 3, 128, 256],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[256], 
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(conv_layer4, weights, strides=[1,1,1,1],padding='SAME')
        conv_biases = tf.nn.bias_add(conv, biases)
        conv_layer5 = tf.nn.relu(conv_biases, name = 'conv5')
    
    # Convolution_layer6
    with tf.variable_scope('convolution_layer6') as scope:
        weights = tf.get_variable('weights',
                                  shape=[3, 3, 256, 256],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[256], 
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(conv_layer5, weights, strides=[1, 1, 1, 1], padding='SAME')
        conv_biases = tf.nn.bias_add(conv, biases)
        conv_layer6 = tf.nn.relu(conv_biases, name = 'conv6')
    
    
    # Maxpooling1_layer6
    with tf.variable_scope('maxpooling1_layer6') as scope:
        maxpool6 = tf.nn.max_pool(conv_layer6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME',name='pooling6')
    
        #maxpool6 = tf.nn.dropout(maxpool6, dropout)
    
    
    # Fullconnected_layer7
    with tf.variable_scope('fullconnected_layer7') as scope:
        reshape = tf.reshape(maxpool6, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights',
                                  shape=[dim, 256],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[256],
                                 dtype=tf.float32, 
                                 initializer=tf.constant_initializer(0.1))
        fullconnected_layer7 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name= "full7")    
    
    
    # Fullconnected_layer8
    with tf.variable_scope('fullconnected_layer8') as scope:
        weights = tf.get_variable('weights',
                                  shape=[256, 256],
                                  dtype=tf.float32, 
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[256],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        fullconnected_layer8 = tf.nn.relu(tf.matmul(fullconnected_layer7, weights) + biases, name="full8")
     
        
    # Softmax
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('softmax_linear',
                                  shape=[256, classes],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32))
        biases = tf.get_variable('biases', 
                                 shape=[classes],
                                 dtype=tf.float32, 
                                 initializer=tf.constant_initializer(0.1))
        softmax_linear = tf.add(tf.matmul(fullconnected_layer8, weights), biases, name='softmax_linear')
    
    return softmax_linear


def losses(logits, labels):
    """Compute loss from logits and labels.
    
    Args:
        logits: logits tensor [batch_size, label_of_predict]
        labels: label tensor [label_of_groudtruth]
        
    Returns:
        loss: loss tensor
    """
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, 
                                                                       labels=labels, name='cross_entropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name + '/loss', loss)
    return loss


def trainning(loss, learning_rate):
    """Training ops.
    
    Args:
        loss: loss tensor, from losses()
        learing_rate: learning rate of optimizer
    Returns:
        train_op: The op for trainning     
    """
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step= global_step)
    return train_op


def evaluation(logits, labels):
  """Evaluate accurracy of predicting image of label.
  
  Args:
    logits: Logits tensor, [batch_size, label_of_predict].
    labels: Labels tensor, [label_of_groudtruth].
  Returns:
    accuracy: Tensor with the number of examples that were predicted correctly.
  """
  with tf.variable_scope('accuracy') as scope:
      correct = tf.nn.in_top_k(logits, labels, 1)
      correct = tf.cast(correct, tf.float16)
      accuracy = tf.reduce_mean(correct)
      tf.summary.scalar(scope.name+'/accuracy', accuracy)
  return accuracy






