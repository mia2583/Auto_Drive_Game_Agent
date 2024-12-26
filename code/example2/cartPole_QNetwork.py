import numpy as np
import tensorflow as tf

import gym
env = gym.make('CartPole-v1', render_mode='human')

input_size = env.observation_space.shape[0]
output_size = env.action_space.n
learning_rate = 1e-1
dis = 0.9
num_episodes = 2000
rList = []

# xavier 초기화
W1 = tf.Variable(
        tf.keras.initializers.GlorotUniform()(shape=[input_size, output_size]),
        name="W1"
)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)


for i in range(num_episodes):
    s = env.reset()[0]
    e = 0.1 / (i+1)
    rAll = 0
    step_count = 0
    done = False

    while not done:
        step_count += 1
        x = np.reshape(s, [1, input_size]).astype(np.float32)
        
        with tf.GradientTape() as tape:
            Qpred = tf.matmul(x, W1)

            # e-greedy
            if np.random.rand(1) < e:
                a = env.action_space.sample()
            else:
                a = tf.argmax(Qpred[0]).numpy()

            new_state, reward, done, truncated, info = env.step(a)
            
            Qs = Qpred.numpy()
            if done:
                Qs[0, a] = -100
            else:
                new_x = np.reshape(new_state, [1, input_size]).astype(np.float32)
                Qnexts = tf.matmul(new_x, W1)
                Qs[0, a] = reward + dis * np.max(Qnexts)

            loss = tf.reduce_sum(tf.square(Qs - Qpred))

        grads = tape.gradient(loss, [W1])
        optimizer.apply_gradients(zip(grads, [W1]))

        rAll += reward
        s = new_state
    rList.append(step_count)
    print("Epissode: {} steps: {}".format(i, step_count))

    if len(rList) > 10 and np.mean(rList[-10:]) > 500:
        print("Solved!")
        break

observation = env.reset()
reward_sum = 0
while True:
    env.render()
    x = np.reshape(observation, [1, input_size]).astype(np.float32)
    Qs = tf.matmul(x, W1).numpy()
    a = np.argmax(Qs)

    observation, reward, done, truncated, info = env.step(a)
    reward_sum += reward
    if done:
        print("Total score: {}".format(reward_sum))
        break

env.close()
