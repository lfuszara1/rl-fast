import numpy as np
import gym
import tensorflow as tf

import random

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


env = gym.make('LunarLander-v2')
num_actions = env.action_space.n
num_observ = env.observation_space.shape
best_result = 200.0
episodes = 1000
scores = []
update_every = 1
epsilon = 0.01
optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)


class Attended(tf.keras.Model):
    def __init__(self, num_attention):
        super(Attended, self).__init__(name='')
        
        self.num_attention = num_attention

        self.attentions = []
        for _ in range(num_attention):
            dense1 = tf.keras.layers.Dense(128, input_dim=128, kernel_regularizer=tf.keras.regularizers.l2(0.001))
            lrelu1 = tf.keras.layers.LeakyReLU()
            dropout1 = tf.keras.layers.Dropout(0.5)
            dense2 = tf.keras.layers.Dense(128, input_dim=128, kernel_regularizer=tf.keras.regularizers.l2(0.001))
            lrelu2 = tf.keras.layers.LeakyReLU()
            dropout2 = tf.keras.layers.Dropout(0.5)
            attention = tf.keras.layers.Attention(128)
            
            self.attentions.append([dense1, lrelu1, dropout1, dense2, lrelu2, dropout2, attention])

        self.softmax = tf.keras.layers.Dense(num_actions, input_dim=128, activation='softmax')

    def call(self, inputs):

        x1 = []
        x2 = []
        xa = []
        for i, j in enumerate(self.attentions):
            if i == 0:
                x1.append(j[0](inputs))
            else:
                x1.append(j[0](xa[-1]))
            x1.append(j[1](x1[-1]))
            x1.append(j[2](x1[-1]))
            x2.append(j[3](x1[-1]))
            x2.append(j[4](x2[-1]))
            x2.append(j[5](x2[-1]))
            xa.append(j[6]([x1[-1], x2[-1]]))

        out = self.softmax(xa[-1])

        return out


model = Attended(1)
model.build((None, num_observ[0]))

model.load_weights('./checkpoints/attended')

counter = 0
for e in range(episodes):
    done = False
    while not done:
        s = env.reset()
        s = s.reshape([1, num_observ[0]])

        r = model(s)
        a_dist = r.numpy()

        a = np.random.choice(a_dist[0], p=a_dist[0])
        a = np.argmax(a_dist == a)

        s, r, done, _ = env.step(a)
        env.render()
        counter += r


env.close()

print("Episodes {} Total score {}".format(episodes, float(counter) / episodes))
