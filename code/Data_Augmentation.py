import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# 상수 설정
action1_path_dir = "./data/discrete/action_1/"
action1_list = os.listdir(action1_path_dir)
action2_path_dir = "./data/discrete/action_2/"
action2_list = os.listdir(action2_path_dir)

# action_1을 flip해서 action_2에 저장
for f in action1_list:
    path = "./data/discrete/action_1/" + f
    load_observations = Image.open(path)
    flip_observations = load_observations.transpose(Image.FLIP_LEFT_RIGHT)
    flip_observations.save("./data/discrete/action_2/" + f )
print("Done action1 -> action2")

# action_2을 flip해서 action_1에 저장
for f in action2_list:
    # action1->action2를 다시 뒤집지 않기 위해서 무시
    if(f[-11:-4] == "action1") : continue
    # 그 외 파일은 좌우반전 후 저장 
    path = "./data/discrete/action_2/" + f
    load_observations = Image.open(path)
    flip_observations = load_observations.transpose(Image.FLIP_LEFT_RIGHT)
    flip_observations.save("./data/discrete/action_1/" + f )
print("Done action2 -> action1")
    