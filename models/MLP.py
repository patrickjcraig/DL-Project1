import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.layers import Dropout
from tensorflow.keras.initializers import GlorotUniform

def create_mlp(input_shape, num_classes, layer_units=None, activation='relu', learning_rate=0.001, optimizer='adam', dropout_rate=0.0):
    if layer_units is None:
        layer_units = {i: 128 for i in range(2)}  # Default to 2 hidden layers with 128 units each
    
    model = Sequential()
    model.add(Input(shape=(input_shape)))
    model.add(Flatten())
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

    if __name__ == "__main__":
        input_shape = (28,28,1)  # Example for flattened 28x28 images
        num_classes = 10   # Example for 10 classes (e.g., digits 0-9)
        layer_units = {0: 256, 1: 128}  # Example layer configuration

        model = create_mlp(input_shape, num_classes, layer_units, activation='relu', learning_rate=0.001, optimizer='adam', dropout_rate=0.5)
        model.summary()