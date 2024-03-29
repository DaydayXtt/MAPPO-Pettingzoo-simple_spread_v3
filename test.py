import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from train import Runner_MAPPO_MPE
from PIL import Image

rewardpath = 'D:/RL/MAPPO/mine-mappo/data_train/'
resultpath = 'D:/RL/MAPPO/mine-mappo/result1/'
if not os.path.exists(resultpath):
    os.makedirs(resultpath)

parser = argparse.ArgumentParser("Hyperparameters Setting for MAPPO in MPE environment")
parser.add_argument("--max_train_steps", type=int, default=int(3e6), help=" Maximum number of training steps")
parser.add_argument("--episode_limit", type=int, default=25, help="Maximum number of steps per episode")
parser.add_argument("--evaluate_freq", type=float, default=5000, help="Evaluate the policy every 'evaluate_freq' steps")
parser.add_argument("--evaluate_times", type=float, default=3, help="Evaluate times")

parser.add_argument("--batch_size", type=int, default=32, help="Batch size (the number of episodes)")
parser.add_argument("--mini_batch_size", type=int, default=8, help="Minibatch size (the number of episodes)")
parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the rnn")
parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the mlp")
parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
parser.add_argument("--epsilon", type=float, default=0.2, help="GAE parameter")
parser.add_argument("--K_epochs", type=int, default=15, help="GAE parameter")
parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
parser.add_argument("--use_reward_norm", type=bool, default=True, help="Trick 3:reward normalization")
parser.add_argument("--use_reward_scaling", type=bool, default=False,
                    help="Trick 4:reward scaling. Here, we do not use it.")
parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
parser.add_argument("--use_relu", type=float, default=False, help="Whether to use relu, if False, we will use tanh")
parser.add_argument("--use_rnn", type=bool, default=False, help="Whether to use RNN")
parser.add_argument("--add_agent_id", type=float, default=False,
                    help="Whether to add agent_id. Here, we do not use it.")
parser.add_argument("--use_value_clip", type=float, default=False, help="Whether to use value clip.")

parser.add_argument('--reward_path', type=str,
                    default=rewardpath + 'MAPPO_env_simple_spread_v3_number_1_seed_0.npy',
                    help='File path to my rewards file')
parser.add_argument('--result_path', type=str,
                    default=resultpath, help='File path to my result')
parser.add_argument('--render_mode', type=str,
                    default='rgb_array', help='File path to my result')

# 解析命令行参数
args = parser.parse_args()
result_dir = args.result_path


assert os.path.exists(result_dir)
gif_dir = os.path.join(result_dir, 'gif')
if not os.path.exists(gif_dir):
    os.makedirs(gif_dir)
gif_num = len([file for file in os.listdir(gif_dir)])  # current number of gif

runner = Runner_MAPPO_MPE(args, env_name="simple_spread_v3", number=1, seed=0)
runner.agent_n.load_model("simple_spread_v3", number=1, seed=0, step=3000)

agent_num = runner.env.num_agents
# reward of each episode of each agent
episode_rewards = {agent: np.zeros(args.max_train_steps) for agent in runner.env.agents}
for episode in range(5):
    states, infos = runner.env.reset()


    agent_reward = {agent: 0 for agent in runner.env.agents}  # agent reward of the current episode
    frame_list = []  # used to save gif

    for episode_step in range(runner.args.episode_limit):
        states = np.array([states[agent] for agent in states.keys()])
        a_n, _ = runner.agent_n.choose_action(states, evaluate=True)
        # need to transit 'a_n' into dict
        actions = {}
        for i, agent in enumerate(runner.env.agents):
            actions[agent] = a_n[i]

        next_states, rewards, dones, _, _ = runner.env.step(actions)

        frame_list.append(Image.fromarray(runner.env.render()))  # 第二次frame_list=[]时报错
        states = next_states

        for agent_id, reward in rewards.items():  # update reward
            agent_reward[agent_id] += reward

    # env.close()
    message = f'episode {episode + 1}, '
    # episode finishes, record reward
    for agent_id, reward in agent_reward.items():
        episode_rewards[agent_id][episode] = reward
        message += f'{agent_id}: {reward:>4f}; '
    print(message)
    # save gif
    frame_list[0].save(os.path.join(gif_dir, f'out{gif_num + episode + 1}.gif'),
                       save_all=True, append_images=frame_list[1:], duration=1, loop=0)


# 读取 .npy 文件
rewards = np.load(args.reward_path)
# print(np.shape(rewards))  # shape:(601,)
fig, ax = plt.subplots()
x = range(1, np.shape(rewards)[0]+1)
ax.plot(x, rewards, label='agent0_1_2')
ax.legend()
ax.set_xlabel('episode')
ax.set_ylabel('reward')
title = f'evaluate result of mappo'
ax.set_title(title)
plt.show()
