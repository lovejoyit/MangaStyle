#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import tensorflow as tf
import manga_datainput
import manga_model

N_CLASSES = 4
IMAGE_HEIGHT = 170
IMAGE_WIDTH = 120
IMAGE_CHANNEL = 3
RATIO = 0.2
BATCH_SIZE = 80
TRAIN_DATA_SIZE = 16000
TEST_DATA_SIZE = 4000
NUM_EPOCHS = 10000
MAX_STEP = 100000 
learning_rate = 0.0001


def run_training():
    """Train model for a number of steps."""
    
    # You need to change the directories to yours.
    train_dir = '//mnt//shared//MangaStyle//train_data.tfrecords'
    test_dir = '//mnt//shared//MangaStyle//test_data.tfrecords'
    logs_train_dir = '///mnt//shared//MangaStyle//logs//train//'
    
    train_batch, train_label_batch = manga_datainput.inputs(BATCH_SIZE, NUM_EPOCHS, train_dir)
    test_batch, test_label_batch = manga_datainput.inputs(BATCH_SIZE, NUM_EPOCHS, test_dir)
    image_batch = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL])
    label_batch = tf.placeholder(tf.int32, shape=[BATCH_SIZE])
    keep_prob = tf.placeholder(tf.float32) 

    logits = manga_model.model(image_batch, BATCH_SIZE, N_CLASSES, keep_prob)
    loss = manga_model.losses(logits, label_batch)        
    train_op = manga_model.trainning(loss, learning_rate)
    acc = manga_model.evaluation(logits, label_batch)

    init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
        
    with tf.Session() as sess:
        saver = tf.train.Saver()
        
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess= sess, coord=coord)
        
        summary_op = tf.summary.merge_all()        
        train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
        epach_count = 0
        
        try:
            for step in np.arange(MAX_STEP):
                if coord.should_stop():
                        break
                
                tra_images,tra_labels = sess.run([train_batch, train_label_batch])
                tra_summary_str, _, tra_loss, tra_acc= sess.run([summary_op, train_op, loss, acc], feed_dict={image_batch:tra_images, label_batch:tra_labels, keep_prob:0.25})
                
                # Print loss and accuracy of training set
                if step % 50 == 0:
                    print('Step %d, training loss = %.2f, training accuracy = %.2f%%' %(step, tra_loss, tra_acc*100.0))
                    train_writer.add_summary(tra_summary_str, step)
                    
                if step % 2000 == 0 or (step + 1) == MAX_STEP:
                    checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step=step)
                    
                # Print accuracy of testing set               
                if step % ((TRAIN_DATA_SIZE / BATCH_SIZE) - 1) == 0 and step > 0:
                    epach_count += 1
                    sum_tes_acc = 0.0
                    # Run for all testing data
                    for test_step in np.arange(TEST_DATA_SIZE / BATCH_SIZE):
                        tes_images,tes_labels = sess.run([test_batch, test_label_batch])
                        tes_loss, tes_acc = sess.run([loss, acc], feed_dict={image_batch:tes_images, label_batch:tes_labels, keep_prob:1.0})
                        sum_tes_acc += tes_acc
                    sum_tes_acc = sum_tes_acc / float((TEST_DATA_SIZE / BATCH_SIZE)) * 100.0
                    print('Epach %d, testing set accuracy = %.2f%%' %(epach_count ,sum_tes_acc))


        except tf.errors.OutOfRangeError:
            print('Done training -- epoch limit reached')
        finally:
            coord.request_stop()           
        coord.join(threads)
        


if __name__ == "__main__":      
        
    run_training()