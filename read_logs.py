import pandas as pd
import matplotlib.pyplot as plt


log_file = './results/pool2D_endreward_finish/2023-05-27T032305_L10/logs/progress.csv'

# Read the log file using pandas
log_data = pd.read_csv(log_file)

# Print the log data
print(log_data)

# Access specific columns from the log data
mean_rewards = log_data["eval/mean_reward"].dropna()

# Print the episode rewards and lengths
print("Episode Rewards:")
print(mean_rewards)


# Plot the training curve
plt.plot(range(len(mean_rewards)), mean_rewards)
plt.xlabel("Timesteps")
plt.ylabel("Eval mean Reward")
plt.title("Training Curve")
plt.savefig('logs.png')
