from csgo_environment import env
import numpy as np
import time


def timing_report(max_decision_steps: int, max_agents_per_team: int):
    my_env = env(num_team_agents=max_agents_per_team, render_mode=None, debug_mode=False)
    my_env.reset()

    total_decision_steps = 0
    total_agents = max_agents_per_team * 2

    start_time = time.time()
    last_print_time = start_time

    # agent_actions = {agent: True for agent in my_env.possible_agents} 
    # for agent in my_env.agent_iter():
    #     observation, reward, termination, truncation, info = my_env.last()
    #     if termination or truncation:
    #         action = None
    #     else:
    #         action = 0 if agent_actions[agent] else 2
    #         agent_actions[agent] = not agent_actions[agent]  # Toggle this agent's next action
            
    #     my_env.step(action)

    #     total_decision_steps += 1
    #     if total_decision_steps >= max_decision_steps:
    #         break

    for agent in my_env.agent_iter():
        observation, reward, termination, truncation, info = my_env.last()

        if all(my_env.terminations.values()) or all(my_env.truncations.values()):
            time1 = time.time()
            my_env.reset()
            continue

        action_mask = info["action_mask"]
        if termination or truncation:
            action = None
        else:
            valid_actions = np.where(action_mask)[0]
            action = np.random.choice(valid_actions)
        my_env.step(action)

        total_decision_steps += 1
        if total_decision_steps >= max_decision_steps:
            break

        # Print stats every second
        current_time = time.time()
        if current_time - last_print_time >= 1.0:
            steps_per_second = total_decision_steps / (current_time - start_time)
            print(f"Steps per second: {steps_per_second:.2f} | Total steps: {total_decision_steps}")
            last_print_time = current_time

    avg_agent_decisions = total_decision_steps / total_agents
    avg_ticks_per_decision = my_env.engine.physics_ticks / avg_agent_decisions
    total_distance = sum(agent.stats.total_distance for agent in my_env.engine.agents.values())
    total_xy_distance = sum(agent.stats.total_xy_distance for agent in my_env.engine.agents.values())
    total_jumps = sum(agent.stats.total_jumps for agent in my_env.engine.agents.values())
    total_distance_per_agent = total_distance / total_agents
    total_xy_distance_per_agent = total_xy_distance / total_agents
    total_jumps_per_agent = total_jumps / total_agents
    real_world_time = my_env.engine.clock.getRealTime()
    physics_time = my_env.engine.game_time
    total_ticks = my_env.engine.physics_ticks
    total_agent_decisions = total_decision_steps
    actual_time_scale = physics_time / real_world_time

    print("\n=== Timing Report ===")
    print(f"Total Agents: {total_agents}")
    print(f"Total Physics Ticks: {total_ticks}")
    print(f"Total Agent Decisions: {total_agent_decisions}")
    print("\nAverages:")
    print(f"  Average Decisions per Agent: {avg_agent_decisions:.2f}")
    print(f"  Average Ticks per Decision per Agent: {avg_ticks_per_decision:.2f}")
    print("\nTiming:")
    print(f"  Real World Time: {real_world_time:.2f} seconds")
    print(f"  Physics Engine Time: {physics_time:.2f} seconds")
    print(f"  Actual Time Scale: {actual_time_scale:.2f}x")
    print("\nAgent Stats:")
    print(f"  Total Distance per Agent: {total_distance_per_agent:.2f}")
    print(f"  Total XY Distance per Agent: {total_xy_distance_per_agent:.2f}")
    print(f"  Total Jumps per Agent: {total_jumps_per_agent:.2f}")
    
    # Aggregate stats across all agents
    all_agents = my_env.engine.agents.values()
    total_cardinal_decisions = sum(agent.stats.total_cardinal_decisions for agent in all_agents)
    total_diagonal_decisions = sum(agent.stats.total_diagonal_decisions for agent in all_agents)
    avg_decision_tick = sum(agent.stats.avg_decision_tick for agent in all_agents) / total_agents
    avg_cardinal_tick = sum(agent.stats.avg_decision_tick_cardinal for agent in all_agents) / total_agents
    avg_diagonal_tick = sum(agent.stats.avg_decision_tick_diagonal for agent in all_agents) / total_agents
    min_cardinal_tick = min(agent.stats.min_cardinal_tick for agent in all_agents)
    max_cardinal_tick = max(agent.stats.max_cardinal_tick for agent in all_agents)
    min_diagonal_tick = min(agent.stats.min_diagonal_tick for agent in all_agents)
    max_diagonal_tick = max(agent.stats.max_diagonal_tick for agent in all_agents)
    total_stuck_count = sum(agent.stats.stuck_count for agent in all_agents)

    print("\nDetailed Movement Stats (All Agents):")
    print(f"  Decision Ticks:")
    print(f"    Average: {avg_decision_tick:.2f}")
    print(f"\n  Cardinal Movements:")
    print(f"    Total Decisions: {total_cardinal_decisions}")
    print(f"    Average Ticks: {avg_cardinal_tick:.2f}")
    print(f"    Min Ticks: {min_cardinal_tick}")
    print(f"    Max Ticks: {max_cardinal_tick}")
    print(f"\n  Diagonal Movements:")
    print(f"    Total Decisions: {total_diagonal_decisions}")
    print(f"    Average Ticks: {avg_diagonal_tick:.2f}")
    print(f"    Min Ticks: {min_diagonal_tick}")
    print(f"    Max Ticks: {max_diagonal_tick}")
    print(f"  Total Stuck Count: {total_stuck_count}")
    print("==================\n")

    my_env.close()


if __name__ == "__main__":
    for max_agents_per_team in [1, 3, 5, 10, 50]:
    # for max_agents_per_team in [1]:
        timing_report(max_decision_steps=10000, max_agents_per_team=max_agents_per_team)