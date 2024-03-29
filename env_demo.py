from pettingzoo.mpe import simple_spread_v3
"""
黑色是landmark；蓝色是agent

observations(N=3): [self_vel,       dim:2 
    self_pos,                       dim:2
    landmark_rel_positions,         dim:2*3
    other_agent_rel_positions,      dim:2*(3-1)
    communication]                  dim:2*(3-1)
    
states:18*N=54

修改各种observation的信息：simple_spread.py中的137行

landmark会动是因为地图缩放，不是landmark的坐标在变
"""
env = simple_spread_v3.env(N=3, max_cycles=5000, local_ratio=0.5, render_mode="human")
env.reset(seed=42)

for agent in env.agent_iter():
    print(agent)
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        action = None
    else:
        # this is where you would insert your policy
        action = env.action_space(agent).sample()

    env.step(action)
env.close()
