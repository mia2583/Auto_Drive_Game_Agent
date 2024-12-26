import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import frozenLake_setting

env = frozenLake_setting.init()

input_size = env.observation_space.n
output_size = env.action_space.n
learning_rate = 0.1
dis = 0.99  # discount factor
num_episodes = 2000
rList = []  # episode마다의 총 리워드

# 입력
X = tf.Variable(tf.ones(shape=[1, input_size], dtype=tf.float32))
# 학습 가능한 weight
W = tf.Variable(tf.random.uniform([input_size, output_size], 0, 0.01))
optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

def one_hot(x, size):
    return np.identity(size)[x:x+1]

for i in range(num_episodes):
    s = env.reset()[0]
    e = 1. / ((i / 50) + 10)
    rAll = 0
    done = False

    while not done:
        X.assign(one_hot(s, input_size))
        with tf.GradientTape() as tape:
            Qpred = tf.matmul(X, W)  # [1, 16] * [16, 4] = [1, 4]
            # e-greedy
            if np.random.rand(1) < e:
                a = env.action_space.sample()
            else:
                a = tf.argmax(Qpred[0]).numpy()

            new_state, reward, done, truncated, info = env.step(a)

            if done:
                Qs = Qpred.numpy()
                Qs[0, a] = reward
            else:
                X.assign(one_hot(new_state, input_size))
                Qnexts = tf.matmul(X, W)
                Qs = Qpred.numpy()
                Qs[0, a] = reward + dis * tf.reduce_max(Qnexts)

            # 손실 함수 정의
            loss = tf.reduce_sum(tf.square(Qs - Qpred))

        # 그래디언트 계산 및 가중치 업데이트
        grads = tape.gradient(loss, [W])
        optimizer.apply_gradients(zip(grads, [W]))

        rAll += reward
        s = new_state
    rList.append(rAll)

# 시각화
print("Success rate: " + str(sum(rList) / num_episodes))
plt.bar(range(len(rList)), rList, color="blue")
plt.show()
