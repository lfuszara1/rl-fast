# EDIM (Encoder Decoder Interpreter Manager)

## Architecture

### Encoder

tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.001))

tf.keras.layers.LeakyReLU()

tf.keras.layers.Dropout(0.5)

### Decoder

tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.001))

tf.keras.layers.LeakyReLU()

tf.keras.layers.Dropout(0.5)

### Interpreter

tf.keras.layers.Attention(256)

### Manager

tf.keras.layers.Dense(num_actions, activation='softmax')

## Effects

CartPole-v0 Solved

CartPols-v1 Solved

LunarLander-v2 To many actions for network

## Notice

1 error per 1000 iterations

Effectivity about 99.5%.
