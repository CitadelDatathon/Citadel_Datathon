import json
from numpy import array
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K


with open('chem_2.json') as f:
    chem_data = json.load(f)

with open('drought.json') as f:
    drought_data = json.load(f)

with open('industry_occupation.json') as f:
    indus_data = json.load(f)

with open('earnings.json') as f:
    earning_data = json.load(f)


train_data = []
train_labels = []

test_data = []
test_labels = []

for fip in drought_data:
    for year in drought_data:
        if fip not in chem_data or year not in chem_data[fip]:
            continue
        if fip not in indus_data or year not in indus_data[fip]:
            continue
        if fip not in earning_data or year not in earning_data[fip]:
            continue

        data = []
        data.append(int(fip))
        data.extend(chem_data[fip][year])

        for data123 in indus_data[fip][year]:
            data.append(int(data123))

        for data123 in earning_data[fip][year]:
            if isinstance(data123, str):
                data123  = data123.replace('+', '').replace(',', '').replace('-', '')
            data.append(int(data123))

        label = drought_data[fip][year]

        if int(year) < 2015:
            train_data.append(data)
            train_labels.append(label)
        else:
            test_data.append(data)
            test_labels.append(label)

train_data = array(train_data)
train_labels = array(train_labels)

test_data = array(test_data)
test_labels = array(test_labels)

print(test_data)

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)

train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

print(test_data)
#print(train_data)
#print(test_data)

def get_length(map_data):
    for fip in map_data:
        for year in map_data[fip]:
            return len(map_data[fip][year])

dim = 1 + get_length(indus_data) + get_length(earning_data) + get_length(chem_data)

def build_model():
    model = keras.Sequential([
    keras.layers.Dense(dim , activation=tf.nn.relu,
                       input_shape=(train_data.shape[1],)),
    keras.layers.Dense(dim, activation=tf.nn.relu),
    keras.layers.Dense(6)
    ])
    optimizer = tf.train.RMSPropOptimizer(0.001)
    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
    return model

model = build_model()
EPOCHS = 500

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
class PrintStep(keras.callbacks.Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.test_data
        # loss, acc = self.model.evaluate(x, y, verbose=0)
        # print('Testing loss: {}, acc: {}'.format(loss, acc))

history = model.fit(train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0,
                    callbacks=[early_stop, PrintStep((test_data, test_labels))])

model.summary()

def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error')
    plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
           label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
           label = 'Val loss')
    plt.legend()
    # plt.show()

plot_history(history)




test_predictions = model.predict(test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [1000$]')
plt.ylabel('Predictions [1000$]')
plt.axis('equal')
plt.xlim(plt.xlim())
plt.ylim(plt.ylim())
plt.plot([-100, 100], [-100, 100])
# plt.show()


error = test_predictions - test_labels.flatten()
std = error.std(axis=0)
print(std)

plt.hist(error, bins = 50)
plt.xlabel("Prediction Error [1000$]")
plt.ylabel("Count")
# plt.show()



f = K.function([model.layers[0].input, K.learning_phase()],
               [model.layers[-1].output])
print(f)