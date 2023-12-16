# IMPORTS
import os
import time
import json
from contextlib import redirect_stdout
from io import StringIO
from functools import partial

import tensorflow as tf
from tensorflow import keras

################################################################

# FOLDER SETUP
# NOTE the folder structure of the original dataset obtained from 
DATASET_PORTION = 'full'
base_dir = f"{os.getcwd()}/dataset/{DATASET_PORTION}"
train_dir = f"{base_dir}/train_and_validation/train"
valid_dir = f"{base_dir}/train_and_validation/validation"

tf.random.set_seed(42)
image_size=(256, 256)
batch_size = 32

# Load data and split into train, validation, and test sets
ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='categorical',
    color_mode='rgb',
    batch_size=batch_size,
    image_size=image_size,
    shuffle=True,
    seed=42,
    validation_split=0.1,
    subset='training',
)

ds_validation = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='categorical',
    color_mode='rgb',
    batch_size=batch_size,
    image_size=image_size,
    shuffle=True,
    seed=42,
    validation_split=0.1,
    subset='validation',
)

ds_test = tf.keras.preprocessing.image_dataset_from_directory(
    valid_dir,
    labels='inferred',
    label_mode='categorical',
    color_mode='rgb',
    batch_size=batch_size,
    image_size=image_size,
    shuffle=False,
    seed=42,
)

input_shape = (256,256,3)
num_classes = 38

# SCALE DATA TO BE BETWEEN 0 AND 1
rescale_layer = tf.keras.layers.Rescaling(scale=1.0 / 255.0)

ds_train_scaled = ds_train.map(lambda x, y: (rescale_layer(x), y))
ds_validation_scaled = ds_validation.map(lambda x, y: (rescale_layer(x), y))
ds_test_scaled = ds_test.map(lambda x, y: (rescale_layer(x), y))

# Check that scaling worked
for batch in ds_train_scaled:
    assert 0<= batch[0][0][0][0][0] and batch[0][0][0][0][0] <=1
    break

################################################################
# INSTANTIATE MODELS
################################################################
# STANDARD NEURAL NETWORK
plane_nn = keras.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax'),
    ]
)

# STANDARD CNN 1
plane_cnn_1 = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        tf.keras.layers.Conv2D(32, 3, padding='valid', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax'),
    ]
)

# STANDARD CNN 2
DefaultConv2D = partial(tf.keras.layers.Conv2D, kernel_size=3, padding="same",
                        activation="relu", kernel_initializer="he_normal")
plane_cnn_2 = tf.keras.Sequential([
    DefaultConv2D(filters=64, kernel_size=7, input_shape=input_shape),
    tf.keras.layers.MaxPool2D(),
    DefaultConv2D(filters=128),
    DefaultConv2D(filters=128),
    tf.keras.layers.MaxPool2D(),
    DefaultConv2D(filters=256),
    DefaultConv2D(filters=256),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=64, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=num_classes, activation="softmax")
])

# INCEPTIONV3
base_model_1 = tf.keras.applications.inception_v3.InceptionV3(weights='imagenet', include_top=False)
# add compatibility with 38 class output layer
x = base_model_1.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
inception_v3 = tf.keras.models.Model(inputs=base_model_1.input, outputs=predictions)

# XCEPTION WITH TUNING
base_model_2 = tf.keras.applications.xception.Xception(weights="imagenet", include_top=False)
avg = tf.keras.layers.GlobalAveragePooling2D()(base_model_2.output)
output = tf.keras.layers.Dense(num_classes, activation="softmax")(avg)
xception = tf.keras.Model(inputs=base_model_2.input, outputs=output)

# freezing the pretrained weights
for layer in base_model_2.layers:
    layer.trainable = False

# unfreezing some of the top layers
for layer in base_model_2.layers[56:]:
    layer.trainable = True

# EFFICIENTNET WITH TUNING
base_model_3 = tf.keras.applications.efficientnet.EfficientNetB1(weights="imagenet", include_top=False)
avg = tf.keras.layers.GlobalAveragePooling2D()(base_model_3.output)
output = tf.keras.layers.Dense(num_classes, activation="softmax")(avg)
efficientnet = tf.keras.Model(inputs=base_model_3.input, outputs=output)

# freezing the pretrained weights
for layer in base_model_3.layers:
    layer.trainable = False

# unfreezing some of the top layers
for layer in base_model_3.layers[56:]:
    layer.trainable = True

metrics = ['accuracy',
           tf.keras.metrics.Recall(),
           tf.keras.metrics.Precision()]

################################################################

# COMPILING MODELS    
plane_nn.compile(
    loss=keras.losses.CategoricalCrossentropy(),
    optimizer='sgd',
    metrics=metrics,
)

plane_cnn_1.compile(
    loss=keras.losses.CategoricalCrossentropy(),
    optimizer=keras.optimizers.legacy.Adam(learning_rate=0.001),
    metrics=metrics,
)

plane_cnn_2.compile(
    loss=keras.losses.CategoricalCrossentropy(),
    optimizer=keras.optimizers.legacy.Adam(learning_rate=0.001),
    metrics=metrics,
)

inception_v3.compile(
    loss='categorical_crossentropy',
    optimizer='rmsprop',
    metrics=metrics,
)

xception.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=0.1, momentum=0.9),
    metrics=metrics)

efficientnet.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.legacy.SGD(learning_rate=0.1, momentum=0.9),
    metrics=metrics)

################################################################

# Setting up fit parameters
fit_params = {
    'steps_per_epoch': 100,
    'epochs': 10,
    'verbose': 2
}

# allow for faster module runtime to ensure it can run
# (i.e. ignore model performance when error checking is True)
error_checking = False
if error_checking:
    fit_params = {
    'steps_per_epoch': 10,
    'epochs': 1,
    'verbose': 2
}
################################################################
def train_and_evaluate_model(model, train_generator, validation_generator, fit_params, model_name):
    """
    Trains and evaluates a given model using the provided data generators and fit parameters.
    
    Args:
        model (keras.Model): The model to be trained and evaluated.
        train_generator (keras.utils.Sequence): The data generator for training data.
        validation_generator (keras.utils.Sequence): The data generator for validation data.
        fit_params (dict): Additional parameters to be passed to the `fit` method of the model.
        model_name (str): The name of the model.
    
    Returns:
        tuple: A tuple containing the performance data dictionary and the training history.
    """
        
    
    # Train the model
    start_time = time.time()
    history = model.fit(train_generator, validation_data=validation_generator, **fit_params)
    end_time = time.time()
    formatted_total_time = end_time - start_time

    # Evaluate the model
    eval_results = model.evaluate(ds_test)

    # Packaging results in a dictionary
    performance_data = {
        'model_name': model_name,
        'training_time': formatted_total_time,
        'loss': eval_results[0],
        'accuracy': eval_results[1],
        'recall': eval_results[2],
        'precision': eval_results[3]
    }

    with open(f"./models/summary/{model_name}_summary.txt", "w") as file:
        model_summary_str = None
        buffer = StringIO()
        with redirect_stdout(buffer):
            model.summary()
        model_summary_str = buffer.getvalue()
        buffer.close()
        file.write(model_summary_str)

    # Write results to a file
    with open(f"./models/performance/{model_name}_performance.json", "w") as file:
        json.dump(performance_data, file, indent=4)

    model.save(f'./models/saved/{model_name}.keras')
    return performance_data, history

################################################################

# associated each model with its name
models = {
    "nn": plane_nn,
    "cnn_1": plane_cnn_1,
    "cnn_2": plane_cnn_2,
    "inception_v3": inception_v3,
    "xception": xception,
    "efficientnet": efficientnet,
}

performances = []
model_histories = []

# obtain performance metrics for each model
for model_name, model in models.items():
    performance, history = train_and_evaluate_model(model, ds_train, ds_validation, fit_params, model_name)
    performances.append(performance)
    model_histories.append(history)
################################################################

import matplotlib.pyplot as plt
import pandas as pd

# plot the training history of each model
for model_name, history in zip(models.keys(),model_histories):   
    df = pd.DataFrame(history.history) #TODO possible set
    df.index += 1
    df.plot(
        figsize=(8,5), xlim=[1,10], ylim=[0,1.5], grid=True, xlabel='Epoch',
        style=['r--','r--','b-','b-*'])
    plt.savefig(f"./models/history/{model_name}_history_plot.jpg")
    