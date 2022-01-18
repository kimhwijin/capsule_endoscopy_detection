import base64
from random import sample
import tensorflow as tf
from tensorflow import keras
import os
import json
import matplotlib.pyplot as plt
import base64
import numpy as np
import csv
from glob import glob


dataset_path = os.getcwd()
dataset_path = os.path.join(dataset_path, 'dataset')

train_json_paths = sorted(glob(os.path.join(dataset_path, 'train', '*.json')))
test_json_paths = sorted(glob(os.path.join(dataset_path, 'test', '*.json')))
train_json_paths = tf.constant(train_json_paths)
test_json_paths = tf.constant(test_json_paths)

print('train, test : ' ,len(train_json_paths), len(test_json_paths))

classes = []
labels = []
with open(os.path.join(dataset_path, 'class_id_info.csv')) as class_id_info:
    class_to_id = csv.reader(class_id_info)
    for row in class_id_info:
        c, l = row.strip().split(',')
        classes.append(c)
        labels.append(l)

_classes = tf.constant(classes[1:])
_labels = tf.constant(list(map(int, labels[1:])))

print('class name, label : ' , _classes.numpy(), _labels.numpy())

def decode_base64_image_sample():
    with open(train_json_paths[0].numpy(), 'r') as sample_json:
        sample_json = json.load(sample_json)

    image_data = sample_json['imageData']
    image = base64.b64decode(image_data)
    image = tf.image.decode_image(image, channels=3)
    image = tf.cast(image, dtype=tf.float32)

    plt.imshow(image / 255.)

def decode_tensorflow_compute_base64_image_sample():
    with open(train_json_paths[3].numpy(), 'r') as sample_json:
        sample_json = json.load(sample_json)

    image_data = sample_json['imageData']
    image_data = tf.strings.regex_replace(image_data, '[/]', '_')
    image_data = tf.strings.regex_replace(image_data, '[+]', '-')
    image = tf.io.decode_base64(image_data)

    image = tf.image.decode_image(image, channels=3)
    image = tf.cast(image, dtype=tf.float32) / 255.

    plt.imshow(image)


def print_sample_json(sample_json):
    print(sample_json.keys())
    print(sample_json['shapes'][0].keys())
    print(sample_json['shapes'][0]['label'])
    print(sample_json['shapes'][0]['points'])
    print(sample_json['imageHeight'], sample_json['imageWidth'])
    point = np.array(sample_json['shapes'][0]['points'])
    point[:, 1] = point[:, 1] / sample_json['imageHeight']
    point[:, 0] = point[:, 0] / sample_json['imageWidth']
    print(point)

#1
def load_all_json_data_to_list(json_path):
    base64_images = []
    points = []
    class_names = []
    for i, json_path in enumerate(json_path):
        with open(json_path.numpy(), 'r') as _json:
            data = json.load((_json))

            base64_images.append(data['imageData'])
            
            point = np.array(data['shapes'][0]['points'])
            point[:, 1] = point[:, 1] / data['imageHeight']
            point[:, 0] = point[:, 0] / data['imageWidth']
            points.append(point)
            
            class_names.append(data['shapes'][0]['label'])
    return base64_images, points, class_names


def points_to_bounding_box(point):
    
    offset_width, offset_height = point[0]
    box_width = point[2][0] - point[0][0]
    box_height = point[2][1] - point[0][1]
    point = list(map(int, [offset_height, offset_width, box_height, box_width]))
    return point

#2
def make_data_list_to_tensor(base64_images, points, class_names):

    box_points = list(map(points_to_bounding_box, points))
    tf_base64_images = tf.data.Dataset.from_tensor_slices(tf.constant(base64_images))
    tf_points = tf.data.Dataset.from_tensor_slices(tf.constant(box_points))
    tf_class_names = tf.data.Dataset.from_tensor_slices(tf.constant(class_names))

    return tf_base64_images, tf_points, tf_class_names


def decode_image(base64_image, point, class_name):
    
    base64_image = tf.strings.regex_replace(base64_image, '[/]', '_')
    base64_image = tf.strings.regex_replace(base64_image, '[+]', '-')

    image = tf.io.decode_base64(base64_image)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    
    image = tf.image.resize(image, [299, 299], method='nearest')
    image = tf.cast(image, dtype=tf.float32) / 255.
    
    return image, point, class_name

def class_to_label(image, point, class_name):
    label = _labels[tf.equal(_classes, class_name)]
    return image, point, label

def extract_labels(image, point, label):
    return image, (label, point)


#3
def create_dataset(tf_base64_images, tf_points, tf_class_names, autotune=tf.data.AUTOTUNE, shuffle_buffer_size=None, batch_size=32, cache=True):
    dataset = tf.data.Dataset.zip((tf_base64_images, tf_points, tf_class_names))
    dataset = dataset.map(decode_image, num_parallel_calls=autotune)
    dataset = dataset.map(class_to_label, num_parallel_calls=autotune)
    dataset = dataset.map(extract_labels, num_parallel_calls=autotune)
    if cache:
        dataset = dataset.cache()

    if shuffle_buffer_size:
        dataset = dataset.shuffle(shuffle_buffer_size)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(autotune)
    
    return dataset