import gym
from stable_baselines3 import DQN

# Load the CartPole environment
env = gym.make("CartPole-v1")

# Initialize the DQN model (using a multi-layer perceptron policy)
model = DQN(
    "MlpPolicy", 
    env, 
    verbose=1, 
    learning_rate=1e-4,  # Experiment with lower learning rate
    batch_size=64, 
    buffer_size=100000,  # Larger replay buffer
    target_update_interval=1000,  # Increase target network update frequency
    exploration_fraction=0.2,  # Exploration settings
    exploration_final_eps=0.02,
)


# Train the model for a certain number of timesteps (e.g., 10,000)
model.learn(total_timesteps=500000)

# Save the trained model for future use
model.save("dqn_cartpole")

# Close the environment
env.close()

print("Model training complete and saved as 'dqn_cartpole'.")
