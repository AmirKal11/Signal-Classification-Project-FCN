import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_model(input_dim, h_size=64, depth=6, dropout_rate=0.2):
    """
    Builds a Deep Neural Network using the Keras Functional API.
    This ensures model.summary() works immediately and saving is automatic.
    """
    
    # 1. Define Input Layer
    inputs = keras.Input(shape=(input_dim,), name="input_layer")
    
    # 2. Initial Block
    x = layers.Dense(h_size, name="init_dense")(inputs)
    x = layers.PReLU(name="init_activation")(x)
    x = layers.Dropout(dropout_rate, name="init_dropout")(x)
    
    # 3. Dynamic Hidden Blocks
    for i in range(depth):
        x = layers.Dense(h_size, name=f"hidden_dense_{i}")(x)
        x = layers.PReLU(name=f"hidden_activation_{i}")(x)
        x = layers.Dropout(dropout_rate, name=f"hidden_dropout_{i}")(x)
        
    # 4. Output Layer
    outputs = layers.Dense(1, activation='sigmoid', name="output_layer")(x)
    
    # 5. Instantiate Model
    model = keras.Model(inputs=inputs, outputs=outputs, name="DiTauClassifier")
    
    # 6. Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model