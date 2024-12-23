import numpy as np
import matplotlib.pyplot as plt
import random
import frozenLake_setting

"""주어진 배열에서 최댓값을 가진 요소 중 랜덤으로 인덱스 반환"""
def rargmax(vector):
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0] # 최대값과 같은 요소에 True, 나머지는 False한 1차원 배열
    return random.choice(indices)

env = frozenLake_setting.init(is_slippery=False)

# Q 테이블 생성
Q = np.zeros([env.observation_space.n, env.action_space.n])
num_episodes = 2000
# episode마다의 총 리워드
rList = []

for i in range(num_episodes):
    state = env.reset()[0]
    rAll = 0 # 리워드 총합
    done = False

    while not done:
        action = rargmax(Q[state, :])

        new_state, reward, done, truncated, info = env.step(action)
        
        Q[state, action] = reward + np.max(Q[new_state, :])
        
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
