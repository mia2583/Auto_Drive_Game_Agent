import numpy as np
import matplotlib.pyplot as plt
import frozenLake_setting

env = frozenLake_setting.init()

# Q 테이블 생성
Q = np.zeros([env.observation_space.n, env.action_space.n])
learning_rate = 0.85
dis = 0.99 # discount factor
num_episodes = 2000
# episode마다의 총 리워드
rList = []

for i in range(num_episodes):
    state = env.reset()[0]
    rAll = 0 # 리워드 총합
    done = False
    e = 0.1 / (i+1) # e-greedy factor

    while not done:
        # e-greedy
        if np.random.rand(1) < e:
            action = env.action_space.sample()
        else:
            # Q(a,s)에 random noise 추가(decaying)
            action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) / (i+1))

        new_state, reward, done, truncated, info = env.step(action)

        # discount future reward(decay rate)
        # non deterministic을 극복하기 위해 learning_rate 사용
        Q[state, action] = (1-learning_rate) * Q[state, action] \
            + learning_rate * (reward + dis * np.max(Q[new_state, :]))
        
        rAll += reward
        state = new_state

    rList.append(rAll)
    
#최종 Q 테이블 출력
print("Final Q-Table Values")
print("LEFT DOWN RIGHT UP")
print(Q)
    
#시각화
print("Success rate: " + str(sum(rList)/num_episodes))
plt.bar(range(len(rList)), rList, color="blue")
plt.show()
