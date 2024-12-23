import frozenLake_setting

env = frozenLake_setting.init(is_slippery=False)
env.reset()
print(env.render())

while True:
	key = frozenLake_setting.input_key()

	if key not in frozenLake_setting.arrow_keys:
		print("Game aborted!")
		break
	
	action = frozenLake_setting.arrow_keys[key]
	state, reward, done, _, info = env.step(action)
	print("State: ", state, ", Action: ", action, ", Reward: ", reward, ", Info: ", info)
	print(env.render())

	if done:
		print("Finished with reward ", reward)
		break
	