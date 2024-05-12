import numpy as np
from skimage.transform import resize
from skimage.color import rgb2gray
from config import *

def count_max_lives(environment):
    environment.reset()
    _, _, _, env_info = environment.step(0)
    return env_info['lives']

def check_if_live(life, current_life):
    if life > current_life:
        return True
    else:
        return False

def process_frame(raw_observation):
    processed_observation = np.uint8(resize(rgb2gray(raw_observation), (input_height, input_width), mode='reflect') * 255)
    return processed_observation

def get_initialization_state(history_buffer, state, history_length):
    for i in range(history_length):
        history_buffer[i, :, :] = process_frame(state)
