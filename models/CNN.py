import tensorflow as tf
from tensorflow.keras import layers, models

def create_cnn_model(input_shape, num_classes, hyperparameters):
    model = models.Sequential()
    
    # Add convolutional layers based on hyperparameters
    for i in range(hyperparameters['num_layers']):
        filters = hyperparameters['filters'][i]
        kernel_size = hyperparameters['kernel_sizes'][i]
        activation = hyperparameters['activations'][i]
        
        if i == 0:
            model.add(layers.Conv2D(filters, kernel_size, activation=activation, input_shape=input_shape))
        else:
            model.add(layers.Conv2D(filters, kernel_size, activation=activation))
        
        model.add(layers.MaxPooling2D((2, 2)))
    
    # Flatten the output and add dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(hyperparameters['dense_units'], activation=hyperparameters['dense_activation']))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model

# Example usage
if __name__ == "__main__":
    input_shape = (64, 64, 3)  # Example input shape
    num_classes = 10  # Example number of classes
    hyperparameters = {
        'num_layers': 3,
        'filters': [32, 64, 64],
        'kernel_sizes': [(3, 3), (3, 3), (3, 3)],
        'activations': ['relu', 'relu', 'relu'],
        'dense_units': 64,
        'dense_activation': 'relu'
    }
    model = create_cnn_model(input_shape, num_classes, hyperparameters)
    model.summary()