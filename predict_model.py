import json
from numpy import array
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

with open('chem_2.json') as f:
    chem_data = json.load(f)

with open('drought.json') as f:
    drought_data = json.load(f)


train_data = []
train_labels = []

testing_data = []
testing_labels = []

for fip in drought_data:
    for year in drought_data:
        if fip not in chem_data or year not in chem_data[fip]:
            continue
        data= chem_data[fip][year]
        label = drought_data[fip][year]

        if int(year) < 2015:
            train_data.append(data)
            train_labels.append(label)
        else:
            testing_data.append(data)
            testing_labels.append(label)

train_data = array(train_data)
train_labels = array(train_labels)

print(train_data)
print(train_labels)

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std


def build_model():
    model = keras.Sequential([
    keras.layers.Dense(6, activation=tf.nn.relu,
                       input_shape=(train_data.shape[1],)),
    keras.layers.Dense(6, activation=tf.nn.relu),
    keras.layers.Dense(6)
    ])
    optimizer = tf.train.RMSPropOptimizer(0.001)
    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
    return model

model = build_model()
model.summary()

EPOCHS = 50

# Store training stats
history = model.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0)


def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [1000$]')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
           label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
           label = 'Val loss')
    plt.legend()
    plt.ylim([0, 5])
    plt.show()

plot_history(history)





