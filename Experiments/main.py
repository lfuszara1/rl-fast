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
epsilon = 1.0
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01, clipvalue=1.0)


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
            attentione = tf.keras.layers.Attention(128)
            
            self.attentions.append([dense1, lrelu1, dropout1, dense2, lrelu2, dropout2, attentione])

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


model = Attended(4)
model.build((None, num_observ[0]))

grad_buffer = model.trainable_variables
for ix, grad in enumerate(grad_buffer):
    grad_buffer[ix] = grad * 0

counter = 0
epsilon_gradient = epsilon
for e in range(episodes):
    s = env.reset()
    ep_memory = []
    ep = 0
    ep_score = 1
    done = False
    while not done:
        ep += 1
        env.render()
        s = s.reshape([1, num_observ[0]])
        with tf.GradientTape() as tape:
            logits = model(s)
            a_dist = logits.numpy()
            if epsilon > 0.0 and random.uniform(0.0, 1.0) < epsilon:
                a = random.randint(0, num_actions-1)
            else:
                a = np.random.choice(a_dist[0], p=a_dist[0])
                a = np.argmax(a_dist == a)
            loss = compute_loss([a], logits)
        s, r, done, _ = env.step(a)
        ep_score += r
        grads = tape.gradient(loss, model.trainable_variables)       
        ep_memory.append([grads, ep_score + ep])
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

    epsilon_gradient -= 0.02
env.close()

model.save_weights('./checkpoints/attended')

