import base64
from random import sample
import tensorflow as tf
from tensorflow import keras
from pathlib import Path
import os
import json
import matplotlib.pyplot as plt
import base64
import numpy as np

dataset_path = Path(os.getcwd())
dataset_path = dataset_path / 'dataset' / 'train'
train_set = list(sorted(dataset_path.glob('*.json')))

with open(train_set[0], 'r') as sample_json:
    sample_json = json.load(sample_json)

image_data = sample_json['imageData']
image = base64.b64decode(image_data)
image = tf.image.decode_image(image, channels=3)
image = tf.cast(image, dtype=tf.float32)
plt.figure(figsize=(10,10))
plt.imshow(image / 255.)
import time
while True:
    print(0)
    time.sleep(1)