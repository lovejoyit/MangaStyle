#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import tensorflow as tf
import numpy as np
from PIL import Image 

IMG_W = 120  
IMG_H = 170

def get_list(file_dir, split_ratio):
    '''Get all image directory and labels and suffle the data list.
    Args:
        file_dir: File directory.
        split_ratio: Ratio to split the dataset into testing set.  
    Returns:
        train_image_list_shuffle: Training image directories list.
        train_label_list_shuffle: Training label list.
        test_image_list_shuffle: Testing image directories list.
        test_label_list_shuffle: Testing label list.
    '''
    images_list = []
    subfolders_list = []
    image_labels_list = []
    train_test_count = []
    
    for root, dirs, files in os.walk(file_dir):
        temp_image_list = []
        for name in files:
            temp_image_list.append(os.path.join(root, name))
        for name in dirs:
            subfolders_list.append(os.path.join(root, name))
        if len(temp_image_list):
            
            # Suffle each image name list
            shuffle_list = np.array(temp_image_list)
            np.random.shuffle(shuffle_list)
            images_list += list(shuffle_list)
            
            # Count image number for train and test
            each_sample_count = len(temp_image_list)
            each_test_sample_count = math.floor(float(each_sample_count) * split_ratio + 0.5)
            each_train_sample_count = each_sample_count - each_test_sample_count
            temp_count = (each_train_sample_count, each_test_sample_count)
            train_test_count += temp_count
    
    
    for subfolder in subfolders_list:        
        image_count = len(os.listdir(subfolder))
        folder_name = subfolder.split('/')[-1]
        if folder_name == 'boy':
            image_labels_list = np.append(image_labels_list, image_count * [0])
        elif folder_name == 'girl':
            image_labels_list = np.append(image_labels_list, image_count * [1])
        elif folder_name == 'lady':
            image_labels_list = np.append(image_labels_list, image_count * [2])
        else:
            image_labels_list = np.append(image_labels_list, image_count * [3])
        
    
    data_index = 0
    data_type_count = 0
    train_img_list = [] 
    test_img_list = []
    train_labels_list = []
    test_labels_list = []
    
    # Combine image name list and label list for train and test data
    for data_count in train_test_count:
        if data_type_count % 2 == 0:
            train_img_list += images_list[data_index:data_count + data_index]
            train_labels_list += list(image_labels_list[data_index: data_count+ data_index])
        else:
            test_img_list += images_list[data_index:data_count + data_index]
            test_labels_list += list(image_labels_list[data_index: data_count + data_index])
        data_index += data_count
        data_type_count += 1
    

    # Shuffle whole test list
    collect_test_list = np.array([test_img_list, test_labels_list])
    collect_test_list = collect_test_list.transpose()
    np.random.shuffle(collect_test_list)
    
    test_image_list_shuffle = list(collect_test_list[:, 0])
    test_label_list_shuffle = list(collect_test_list[:, 1])
    test_label_list_shuffle = [int(float(label)) for label in test_label_list_shuffle]
    
    # Shuffle whole train list
    collect_train_list = np.array([train_img_list, train_labels_list])
    collect_train_list = collect_train_list.transpose()
    np.random.shuffle(collect_train_list)

    train_image_list_shuffle = list(collect_train_list[:, 0])
    train_label_list_shuffle = list(collect_train_list[:, 1])
    train_label_list_shuffle = [int(float(label)) for label in train_label_list_shuffle]
    
    
    return train_image_list_shuffle, train_label_list_shuffle, test_image_list_shuffle, test_label_list_shuffle


def int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
     
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    """Wrapper for inserting byte features into Example proto."""
    
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def build_tfrecord(image_list, label_list, filedir, name):
    """Build a tfrecord from images and labels.
    
    Args:
        image_list: List of image directories.
        label_list: List of labels.
        filedir: Directory to save tfrecord.
        name: Name of tfrecord file.
    Returns:
        None.
    """
    tfrecord_name = os.path.join(filedir, name + '.tfrecords')
    samples_count = len(image_list)
    writer = tf.python_io.TFRecordWriter(tfrecord_name)
    
    print('building tfrecord!')
    for index in np.arange(0, samples_count):
        try:
            image = Image.open(image_list[index])
            image = image.resize((IMG_W, IMG_H), Image.LANCZOS)
            image = image.convert('L')
            image = image.convert('RGB')
            image_raw = image.tobytes()
            label = int(label_list[index])
            example = tf.train.Example(features=tf.train.Features(feature={
                            'label':int64_feature(label),
                            'image_raw': bytes_feature(image_raw)}))
            writer.write(example.SerializeToString())
        except IOError as e:
            print('Could not read:', image_list[index])
    writer.close()
    print('building tfrecord complete!')


def read_and_decode(filename_queue):  
    """Read and decode of tfrecord to a image and label.
    
    Args:
        filename_queue: The directory of tfrecord file.
    Returns:
        images: 3D tensor [imagehight, imagewith, channels].
        labels: 1D tensor [imagelabel].
    """
    reader = tf.TFRecordReader()
    _, serialized_data = reader.read(filename_queue)
    image_features = tf.parse_single_example(
                                        serialized_data,
                                        features={
                                               'label': tf.FixedLenFeature([], tf.int64),
                                               'image_raw': tf.FixedLenFeature([], tf.string),
                                               })
    image = tf.decode_raw(image_features['image_raw'], tf.uint8)    
    image = tf.reshape(image, [IMG_H, IMG_W, 3])
    label = tf.cast(image_features['label'], tf.int32)    
    image = tf.cast(image, tf.float32) * (1. / 255)
    return image, label

def inputs(batch_size, num_epochs, filename):
    """Generate batch of images and labels.
    
    Args:
        batch_size: Number of images in each batch.
        num_epochs: Number of epochs limit generate from tfrecord.
        filename: The directory of tfrecord file.
    Returns:
        images: 4D tensor [batch_size, imagehight, imagewith, channels]
        sparse_labels: 1D tensor [label_of_images].
    """
    if not num_epochs: num_epochs = None

    with tf.name_scope('input'):
        filename_queue = tf.train.string_input_producer(
                [filename], num_epochs=num_epochs)

        image, label = read_and_decode(filename_queue)

        images, sparse_labels = tf.train.batch(
                [image, label], batch_size=batch_size, num_threads=1,
                capacity=1000)

    return images, sparse_labels   


if __name__ == "__main__":
    
    # Parameters
    dataset_dir = '//mnt//shared//MangaStyle//dataset' 
    ratio_to_split_dataset = 0.2
    save_tfrecord_dir = '//mnt//shared//MangaStyle//'
    train_dataset_name = 'train_data'
    test_dataset_name = 'test_data'
    
    # Get the list and build tfrecord
    train_image_list, train_label_list, test_image_list, test_label_list = get_list(dataset_dir, ratio_to_split_dataset)
    train_image_list, train_label_list, test_image_list, test_label_list = get_list(dataset_dir, ratio_to_split_dataset)
    build_tfrecord(train_image_list, train_label_list, save_tfrecord_dir, train_dataset_name)
    build_tfrecord(test_image_list, test_label_list, save_tfrecord_dir, test_dataset_name)
   
    
    # Save 2 batch from the training data set from tfrecord,
    # check images are shuffle or not.
    """
    image_reshape = []
    batch_size = 20
    tfrecords_file = save_tfrecord_dir + train_dataset_name + '.tfrecords'
    num_epochs = 2
    image_batch, label_batch = inputs(batch_size, num_epochs, tfrecords_file)
    init_op = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
    with tf.Session()  as sess: 
        
        sess.run(init_op)
        i = 0
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
        try:
            while not coord.should_stop() and i < 2:

                image, label = sess.run([image_batch, label_batch])
                for index in np.arange(0, batch_size):
                    image_reshape = image[index] * 255
                    image_reshape = image_reshape.astype(np.uint8)
                    img=Image.fromarray(image_reshape, 'RGB')
                    img.save(save_tfrecord_dir + 'batch_' + str(i) + '_image_' + str(index) + '_''Label_' + str(label[index]) + '.jpg')
                i += 1
        
        except tf.errors.OutOfRangeError:
            print('done!')
        finally:
            coord.request_stop()
        coord.join(threads)
    """ 
    
    
    
    
    
    
    
    
    
    
    