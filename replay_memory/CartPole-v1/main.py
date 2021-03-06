import numpy as np
import gym
import tensorflow as tf
from collections import deque
import random

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_virtual_device_configuration(gpus[0], [
    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])

compute_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def discount_rewards(r, gamma=0.8):
    discounted_r = np.zeros_like(r)
    running_add = 0.0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add

    discounted_r -= np.mean(discounted_r)
    discounted_r /= np.std(discounted_r)

    return discounted_r


env = gym.make('CartPole-v1')
num_actions = env.action_space.n
num_observ = env.observation_space.shape[0]
best_result = 300.0
episodes = 100000
scores = []
update_every = 1
epsilon = 0.01
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)


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


action_net = Attended()
action_net.build((None, num_observ))
replay_net = Attended()
replay_net.build((None, num_observ))

grad_buffer = action_net.trainable_variables
for ix, grad in enumerate(grad_buffer):
    grad_buffer[ix] = grad * 0

replay_buffer = action_net.trainable_variables
for ix, grad in enumerate(replay_buffer):
    replay_buffer[ix] = grad * 0

memory = deque(maxlen=2000)
counter = 0
for e in range(episodes):
    s = env.reset()
    s = s.reshape([1, num_observ])
    ep_memory = []
    ep_score = 0
    done = False
    while not done:
        env.render()
        with tf.GradientTape() as tape:
            logits = action_net(s)
            a_dist = logits.numpy()
            if random.uniform(0.0, 1.0) < epsilon:
                a = env.action_space.sample()
            else:
                a = np.random.choice(a_dist[0], p=a_dist[0])
                a = np.argmax(a_dist == a)
            loss = compute_loss([a], logits)
        ns, r, done, _ = env.step(a)
        ns = ns.reshape([1, num_observ])
        memory.append([s, ns, a, r, done])
        s = ns
        ep_score += r
        grads = tape.gradient(loss, action_net.trainable_variables)
        ep_memory.append([grads, r])
    scores.append(ep_score)
    ep_memory = np.array(ep_memory)
    ep_memory[:, 1] = discount_rewards(ep_memory[:, 1])

    for grads, r in ep_memory:
        for ix, grad in enumerate(grads):
            grad_buffer[ix] += grad * r

    if len(memory) > 64:
        minibatch = random.sample(memory, 64)
        b_memory = []
        for s, ns, a, r, done in minibatch:
            s = env.reset()
            s = s.reshape([1, num_observ])
            with tf.GradientTape() as tape:
                logits = replay_net(s)
                a_dist = logits.numpy()
                a = np.random.choice(a_dist[0], p=a_dist[0])
                a = np.argmax(a_dist == a)
                loss = compute_loss([a], logits)
            grads = tape.gradient(loss, replay_net.trainable_variables)
            if done:
                b_memory.append([grads, 0.0])
            else:
                b_memory.append([grads, r])
        b_memory = np.array(b_memory)
        b_memory[:, 1] = discount_rewards(b_memory[:, 1])

        for grads, r in b_memory:
            for ix, grad in enumerate(grads):
                replay_buffer[ix] += grad * r

        memory_buffer = grad_buffer + replay_buffer

        if e % update_every == 0:
            optimizer.apply_gradients(zip(memory_buffer, replay_net.trainable_variables))
            optimizer.apply_gradients(zip(memory_buffer, action_net.trainable_variables))
            for ix, grad in enumerate(grad_buffer):
                grad_buffer[ix] = grad * 0

    action_net.trainable_variables[0:-1] += replay_net.trainable_variables[0:-1]

    if e % 10 == 0:
        print("Episode  {}  Score  {} Max {}".format(e, np.mean(scores[-10:]), np.max(scores[-10:])))

    if np.mean(scores[-50:]) >= best_result and scores[-5:][0] >= best_result:
        print("Episode {} Success {}".format(e, scores[-1:][0]))
        counter += 1
    else:
        counter = 0

    if counter == 50:
        break
env.close()

action_net.save_weights('./checkpoints/action_attended')

