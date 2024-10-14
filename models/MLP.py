import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.layers import Dropout
from tensorflow.keras.initializers import GlorotUniform

def create_mlp(input_shape, num_classes, layer_units=None, activation='relu', learning_rate=0.001, optimizer='adam', dropout_rate=0.0):
    if layer_units is None:
        layer_units = {i: 128 for i in range(2)}  # Default to 2 hidden layers with 128 units each
    
    model = Sequential()
    model.add(Dense(layer_units.get(0, 128), activation=activation, input_shape=(input_shape,), kernel_initializer=GlorotUniform(seed=42)))
    model.add(Dropout(dropout_rate))
    for i in range(1, len(layer_units)):
        model.add(Dense(layer_units.get(i, 128), activation=activation, kernel_initializer=GlorotUniform(seed=42)))
        model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))

    # Optimizer Choice
    if optimizer == 'adam':
        opt = Adam(learning_rate=learning_rate)
    elif optimizer == 'sgd':
        opt = SGD(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        opt = RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError("Unsupported optimizer")
    
    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Example usage:
# layer_units = {0: 64, 1: 128, 2: 256}
# model = create_mlp(input_shape=784, num_classes=10, layer_units=layer_units)
# model.summary()