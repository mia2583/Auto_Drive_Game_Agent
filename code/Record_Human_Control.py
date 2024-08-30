import gym
import numpy as np
import pygame
from PIL import Image
import os
####################}######################################################################
# terminated : if environment terminates (eg. due to task completion, failure etc.) (정상종료)
# truncated : if episode truncates due to a time limit or a reason that is not defined as part of the task MDP.(비정상종료)
###########################################################################################
# There are 5 actions: do nothing, steer left, steer right, gas, brake.
#(0,1,2,3,4)

# generate data for train and valid

if __name__ == "__main__":
    action = np.array([0])
    
    def register_input_discrete():
        global quit, restart
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action[0] = 2
                if event.key == pygame.K_RIGHT:
                    action[0] = 1
                if event.key == pygame.K_UP:
                    action[0] = 3
                if event.key == pygame.K_DOWN:
                    action[0] = 4  # set 1.0 for wheels to block to zero rotation
                if event.key == pygame.K_RETURN:
                    restart = True
                if event.key == pygame.K_ESCAPE:
                    quit = True

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    action[0] = 0
                if event.key == pygame.K_RIGHT:
                    action[0] = 0
                if event.key == pygame.K_UP:
                    action[0] = 0
                if event.key == pygame.K_DOWN:
                    action[0] = 0

            elif event.type == pygame.QUIT:
                quit = True

    env = gym.make("CarRacing-v2",continuous=False, render_mode="human")

    quit = False
    epi=0
    
    while not quit:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        step_num=0
        epi+=1
        first_action = False
        while True:
            register_input_discrete()
            state, reward, terminated, truncated, info = env.step(action[0])
            total_reward += reward

            if action[0] != 0:
                first_action =True
            
            if first_action:
                state_image = Image.fromarray(state)
                if os.system('dir -l ./data/discrete/action_{0} | grep ^- | wc -l'.format(epi)) <= 2000:
                    state_image.save("./data/discrete/action_{0}/epi{1}_state{2}_action{3}.png".format(action[0],epi,step_num,action[0]))
                step_num += 1
                
            if steps % 200 == 0 or terminated or truncated:
                print("\naction : " + str(action[0]))
                print(f"step {steps} total_reward {total_reward:+0.2f}")
            steps += 1
            
            if terminated or truncated or restart or quit:
                break
    env.close()