from config import *
from collections import deque
import numpy as np
import random

class ExperienceBuffer(object):
    def __init__(self):
        self.experiences = deque(maxlen=memory_capacity)

    def record(self, state_sequence, action, reward, episode_end):
        self.experiences.append((state_sequence, action, reward, episode_end))

    def get_mini_batch(self, current_step):
        batch = []
        if current_step >= memory_capacity:
            sample_range = memory_capacity
        else:
            sample_range = current_step

        sample_range -= (history_length + 1)
        sample_indices = random.sample(range(sample_range), batch_size)

        for idx in sample_indices:
            sample = []
            for j in range(history_length + 1):
                if idx+j<len(self.experiences):
                    sample.append(self.experiences[idx + j])
                else:
                    sample.append(self.experiences[-1])
            sample = np.array(sample,dtype=object)
            batch.append((np.stack(sample[:, 0], axis=0), sample[3, 1], sample[3, 2], sample[3, 3]))

        return batch

    def __len__(self):
        return len(self.experiences)

    
class SequentialExperienceStorage(ExperienceBuffer):
    """
    Note that while it's technically better to exclude states from previous episodes, in practice this code below works decently.
    """
    def __init__(self):
        super().__init__()

    def get_mini_batch(self, current_timestep):
        batch = []
        if current_timestep >= memory_capacity:
            sample_range = memory_capacity
        else:
            sample_range = current_timestep

        sample_range -= (lstm_sequence_length + 1)
        sample_indices = random.sample(range(sample_range - lstm_sequence_length), batch_size)

        for i in sample_indices:
            sample = []
            for j in range(lstm_sequence_length + 1):
                sample.append(self.experience_log[i + j])
            sample = np.array(sample)
            batch.append((np.stack(sample[:, 0], axis=0), sample[lstm_sequence_length - 1, 1], sample[lstm_sequence_length - 1, 2], sample[lstm_sequence_length - 1, 3]))

        return batch
    