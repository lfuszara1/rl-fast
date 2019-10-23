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


env = gym.make('Breakout-ram-v0')
num_actions = env.action_space.n
num_observ = env.observation_space.shape
best_result = 495.0
episodes = 1000
scores = []
update_every = 1
epsilon = 0.1
gamma = 0.6
optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)


class Attended(tf.keras.Model):
    def __init__(self):
        super(Attended, self).__init__(name='')
        self.dense1 = tf.keras.layers.Dense(512, input_dim=num_observ[0], kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.lrelu1 = tf.keras.layers.LeakyReLU()
        self.dropout1 = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(512, input_dim=512, kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.lrelu2 = tf.keras.layers.LeakyReLU()
        self.dropout2 = tf.keras.layers.Dropout(0.5)

        self.attention = tf.keras.layers.Attention(512)
        self.softmax = tf.keras.layers.Dense(num_actions, input_dim=512, activation='softmax')

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
model.build((None, num_observ[0]))

grad_buffer = model.trainable_variables
for ix, grad in enumerate(grad_buffer):
    grad_buffer[ix] = grad * 0

memory = []
counter = 0
for e in range(episodes):
    s = env.reset()
    ep_memory = []
    ep_score = 0
    done = False
    while not done:
        env.render()
        s = s.reshape([1, num_observ[0]])
        with tf.GradientTape() as tape:
            logits = model(s)
            a_dist = logits.numpy()
            if random.uniform(0.0, 1.0) < epsilon:
                a = random.randint(0, num_actions-1)
            else:
                a = np.argmax(a_dist)
            loss = compute_loss([a], logits)
        s, r, done, _ = env.step(a)
        ep_score += r
        if not done: 
            next_s = s.reshape([1, num_observ[0]])
            next_logits = model(next_s)
            next_a_dist = next_logits.numpy()
            next_a = np.argmax(next_a_dist)
            r += gamma * next_a
        else:
            r -= 10.0

        grads = tape.gradient(loss, model.trainable_variables)
        ep_memory.append([grads, r])
    scores.append(ep_score)
    ep_memory = np.array(ep_memory)
    ep_memory[:, 1] = discount_rewards(ep_memory[:, 1])

    for grads, r in ep_memory:
        for ix, grad in enumerate(grads):
            grad_buffer[ix] += grad * r

    if e % update_every == 0:
        optimizer.apply_gradients(zip(grad_buffer, model.trainable_variables))
        for ix, grad in enumerate(grad_buffer):
            grad_buffer[ix] = grad * 0

    if e % 10 == 0:
        print("Episode  {}  Score  {} Max {}".format(e, np.mean(scores[-10:]), np.max(scores[-10:])))

    if np.mean(scores[-2:]) >= best_result:
        print("Episode {} Success {}".format(e, scores[-1:][0]))
        counter += 1

    if counter == 2:
        break

    if len(memory) > 10000:
        memory.pop(0)
env.close()

model.save_weights('./checkpoints/attended')

