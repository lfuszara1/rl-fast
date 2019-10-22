import numpy as np
import gym
import tensorflow as tf

import time

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0], [
    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

compute_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def discount_rewards(r, gamma=0.8):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


env = gym.make('CartPole-v1')
num_actions = env.action_space.n
num_observ = env.observation_space.shape

episodes = 1000


class Attended(tf.keras.Model):
    def __init__(self):
        super(Attended, self).__init__(name='')

        self.dense1 = tf.keras.layers.Dense(128, input_dim=num_observ, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(128, input_dim=128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.dropout2 = tf.keras.layers.Dropout(0.5)

        self.attention = tf.keras.layers.Attention(128)
        self.softmax = tf.keras.layers.Dense(num_actions, input_dim=128, activation='softmax')

    def call(self, inputs):
        x1 = self.dense1(inputs)
        x1 = self.dropout1(x1)
        x2 = self.dense2(x1)
        x2 = self.dropout2(x2)

        x = self.attention([x1, x2])
        out = self.softmax(x)

        return out


model = Attended()
model.build((None, num_observ[0]))
model.load_weights('./checkpoints/attended')

counter = 0
for e in range(episodes):
    s = env.reset()
    s = s.reshape([1, 4])

    r = model(s)
    a_dist = r.numpy()

    a = np.random.choice(a_dist[0], p=a_dist[0])
    a = np.argmax(a_dist == a)

    s, r, done, _ = env.step(a)
    env.render()
    counter += r

env.close()

print("Episodes {} Total score {}".format(episodes, float(counter) / episodes))
