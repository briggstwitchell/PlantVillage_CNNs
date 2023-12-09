# IMPORTS
import os
import time
import json
from contextlib import redirect_stdout
from io import StringIO

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experiemental.set_memory_growth(physical_devices[0], True)
tf.random.set_seed(42)

# FOLDER SETUP
DATASET_PORTION = 'full'
base_dir = f"{os.getcwd()}/dataset/{DATASET_PORTION}"
train_dir = f"{base_dir}/train_and_validation/train"
valid_dir = f"{base_dir}/train_and_validation/validation"
test_dir = f"{base_dir}/test"

image_size=(256, 256)
batch_size = 64

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

# INSTANTIATE MODELS
input_shape = (256,256,3)
num_classes = 38

# standard neural network
plane_nn = keras.Sequential(
    [
        layers.Flatten(input_shape=input_shape),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax'),
    ]
)

# standard convolutional neural network
plane_cnn = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, 3, padding='valid', activation='relu'),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax'),
    ]
)

# CNN with tuning (extension of googLeNet)
base_model = InceptionV3(weights='imagenet', include_top=False)

# add compatibility with 38 class output layer
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(1024, activation='relu')(x)
predictions = layers.Dense(38, activation='softmax')(x)
inception_v3 = Model(inputs=base_model.input, outputs=predictions)

metrics = ['accuracy',
        #    tf.keras.metrics.F1Score(),
           tf.keras.metrics.Recall(),
           tf.keras.metrics.Precision()]

# COMPILING MODELS    
plane_nn.compile(
    loss=keras.losses.CategoricalCrossentropy(),
    optimizer='sgd',
    metrics=metrics,
)

plane_cnn.compile(
    loss=keras.losses.CategoricalCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    metrics=metrics,
)

inception_v3.compile(
    loss='categorical_crossentropy',
    optimizer='rmsprop',
    metrics=metrics,
)

# SETTING 
fit_params = {
    'steps_per_epoch': 100,
    'epochs': 10,
    'verbose': 2
}

error_checking = True
if error_checking:
    fit_params = {
    'steps_per_epoch': 10,
    'epochs': 1,
    'verbose': 2
}

def format_time(seconds):
    """Convert seconds to Hours:Minutes:Seconds format."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours)}h:{int(minutes)}m:{seconds:.2f}s"


def train_and_evaluate_model(model, train_generator, validation_generator, fit_params, model_name):
    """Fit and obtain test score for the models"""
    # Train the model
    start_time = time.time()
    model.fit(train_generator, validation_data=validation_generator, **fit_params)
    end_time = time.time()
    formatted_total_time = format_time(end_time - start_time)

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

    model.save(f'./models/saved/{model_name}')
    return performance_data

models = {
    "plane_nn": plane_nn,
    "plane_cnn": plane_cnn,
    "inception_v3": inception_v3,
}

performances = []

for model_name, model in models.items():
    performances.append(
        train_and_evaluate_model(model, ds_train, ds_validation, fit_params, model_name)
        )
