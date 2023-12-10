import numpy as np
import time

import keras
import tensorflow as tf
from keras import layers
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tensorflow import data as tf_data


train_ds, validation_ds, test_ds = tfds.load(
    "plant_village",
    # Reserve 10% for validation and 10% for test
    split=["train[:80%]", "train[80%:90%]", "train[90%:100%]"],
    as_supervised=True,  # Include labels
)

print(f"Number of training samples: {train_ds.cardinality()}")
print(f"Number of validation samples: {validation_ds.cardinality()}")
print(f"Number of test samples: {test_ds.cardinality()}")
#################################################################
plt.figure(figsize=(10, 10))
for i, (image, label) in enumerate(train_ds.take(9)):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image)
    plt.title(int(label))
    plt.axis("off")
    if i == 8:
        break
plt.savefig('testplot.png')
#################################################################
resize_fn = keras.layers.Resizing(150, 150)

train_ds = train_ds.map(lambda x, y: (resize_fn(x), y))
validation_ds = validation_ds.map(lambda x, y: (resize_fn(x), y))
test_ds = test_ds.map(lambda x, y: (resize_fn(x), y))
#################################################################
# Function to one-hot encode the labels
def one_hot_encode(image, label):
    num_classes = 38
    return image, tf.one_hot(label, depth=num_classes)

# Apply the one-hot encoding to the datasets
train_ds = train_ds.map(one_hot_encode)
validation_ds = validation_ds.map(one_hot_encode)
test_ds = test_ds.map(one_hot_encode)
#################################################################

augmentation_layers = [
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
]

def data_augmentation(x):
    for layer in augmentation_layers:
        x = layer(x)
    return x

train_ds = train_ds.map(lambda x, y: (data_augmentation(x), y))
##################################################################
# batch the data and use prefetching to optimize loading speed.
batch_size = 64
train_ds = train_ds.batch(batch_size)
validation_ds = validation_ds.batch(batch_size)
test_ds = test_ds.batch(batch_size)
# train_ds = train_ds.batch(batch_size).prefetch(tf_data.AUTOTUNE).cache()
# validation_ds = validation_ds.batch(batch_size).prefetch(tf_data.AUTOTUNE).cache()
# test_ds = test_ds.batch(batch_size).prefetch(tf_data.AUTOTUNE).cache()
##################################################################
# building the base model
base_model = keras.applications.InceptionV3(
    include_top=False,
    weights="imagenet",
    input_shape=(150, 150, 3),
)

# Freeze the base_model
base_model.trainable = False

# Create new model on top
inputs = keras.Input(shape=(150, 150, 3))

# scaling inputs from (0, 255) to a range of (-1., +1.)
scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
x = scale_layer(inputs)

# The base model contains batchnorm layers. We want to keep them in inference mode
# when we unfreeze the base model for fine-tuning, so we make sure that the
# base_model is running in inference mode here.
x = base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
outputs = keras.layers.Dense(38)(x)
model = keras.Model(inputs, outputs)
model.summary(show_trainable=True)

metrics = ['accuracy',
           tf.keras.metrics.Recall(),
           tf.keras.metrics.Precision()]

model.compile(
    loss='categorical_crossentropy',
    optimizer='rmsprop',
    metrics=metrics,
)

epochs = 10
print("Fitting the top layer of the model")
beg_time = time.time()
print(beg_time)
model.fit(train_ds, epochs=epochs, validation_data=validation_ds)
end_time = time.time()
print(end_time)
total_time = end_time - beg_time
print(f'Total train time: {total_time}')

model.evaluate(test_ds)