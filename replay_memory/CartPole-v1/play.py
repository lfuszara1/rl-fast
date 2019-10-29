import numpy as np
import gym
import tensorflow as tf

import random

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0], [
    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])


env = gym.make('CartPole-v1')
num_actions = env.action_space.n
num_observ = env.observation_space.shape[0]
best_result = 300.0
episodes = 1000


class Attended(tf.keras.Model):
    def __init__(self):
        super(Attended, self).__init__(name='')
        self.dense1 = tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.lrelu1 = tf.keras.layers.LeakyReLU()
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.lrelu2 = tf.keras.layers.LeakyReLU()
        self.dropout2 = tf.keras.layers.Dropout(0.5)

        self.attention = tf.keras.layers.Attention(256)

        self.softmax = tf.keras.layers.Dense(num_actions, activation='softmax')

    def call(self, inputs):
        x1 = self.dense1(inputs)
        x1 = self.lrelu1(x1)
        x1 = self.dropout1(x1)
        x2 = self.dense2(x1)
        x2 = self.lrelu2(x2)
        x2 = self.dropout2(x2)

        x = self.attention([x1, x2])

        out = self.softmax(x)

        return out


model = Attended()
model.build((None, num_observ))

model.load_weights('./checkpoints/action_attended')

scores = []
for e in range(episodes):
    s = env.reset()
    ep_score = 0
    done = False
    while not done:
        env.render()
        s = s.reshape([1, num_observ])
        logits = model(s)
        a_dist = logits.numpy()
        a = np.random.choice(a_dist[0], p=a_dist[0])
        a = np.argmax(a_dist == a)
        s, r, done, _ = env.step(a)
        ep_score += r
    scores.append(ep_score)
    print(e, ep_score)

env.close()

print("Episodes {} Total score {}".format(episodes, np.mean(scores)))

