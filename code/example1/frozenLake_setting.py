import gym
from gym.envs.registration import  register

# mecro
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

arrow_keys = {
    'w' : UP,
    'a' : LEFT,
    's' : DOWN,
    'd' : RIGHT
}

def input_key():
    key = input("Enter wasd\n")
    return key

def init():
    register(
	    id = 'FrozenLake-v3',
	    entry_point='gym.envs.toy_text:FrozenLakeEnv',
	    kwargs= {'map_name':'4x4', 'is_slippery':False}
    )
    # render_mode : (['human', 'ansi', 'rgb_array'])
    return gym.make('FrozenLake-v3', render_mode='ansi')

    
    