import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def create_mlp(input_shape, num_classes, layer_units=None):
    if layer_units is None:
        layer_units = {i: 128 for i in range(2)}  # Default to 2 hidden layers with 128 units each
    
    model = Sequential()
    model.add(Dense(layer_units.get(0, 128), activation='relu', input_shape=(input_shape,)))
    
    for i in range(1, len(layer_units)):
        model.add(Dense(layer_units.get(i, 128), activation='relu'))
    
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# Example usage:
# layer_units = {0: 64, 1: 128, 2: 256}
# model = create_mlp(input_shape=784, num_classes=10, layer_units=layer_units)
# model.summary()