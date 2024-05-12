# DQN
num_episodes = 3000
input_height = 84
input_width = 84
history_length = 4
learning_rate_dqn = 0.0001
lstm_sequence_length = 20
evaluation_reward_window = 100
memory_capacity = 1000000
training_frames = 1000
batch_size = 32
gamma_scheduler = 0.4
step_size_scheduler = 100000

# Double DQN
target_update_frequency = 1000
