# main.py - Final GPU + CNN + 双智能体 + UI-B(每20局渲染一次)

import numpy as np
from snake_env import Game
from snake_agent import Agent
from helper import plot


def train():
    plot_scores = []
    plot_mean_scores = []
    record = 0

    # 创建环境（160x160）
    game = Game(width=320, height=320)

    # nC=4 通道（wall, food, snake1, snake2），nA=3（forward/right/left）
    agent1 = Agent(game.nC, game.nA, game.grid_h, game.grid_w)
    agent2 = Agent(game.nC, game.nA, game.grid_h, game.grid_w)

    total_steps = 0

    while True:
        # 初始化一局
        game.reset()
        state1 = game.get_grid_obs(which=1)
        state2 = game.get_grid_obs(which=2)

        done = False
        score = 0

        while not done:
            # 两条蛇各自根据自己的观测选择动作（带 epsilon 探索）
            action1 = agent1.get_action(state1)
            action2 = agent2.get_action(state2)

            # 和环境交互一步
            reward, done, score = game.play_step(action1, action2)

            next_state1 = game.get_grid_obs(which=1)
            next_state2 = game.get_grid_obs(which=2)

            # 双智能体共用 team reward
            agent1.remember(state1, action1, reward, next_state1, done)
            agent2.remember(state2, action2, reward, next_state2, done)

            # 训练
            agent1.train_long_memory(batch_size=512)
            agent2.train_long_memory(batch_size=512)

            state1 = next_state1
            state2 = next_state2

            total_steps += 1

            # 周期性更新 target 网络（提高稳定性）
            if total_steps % 1000 == 0:
                agent1.update_target()
                agent2.update_target()

        # 一局结束
        agent1.n_game += 1
        agent2.n_game += 1

        # 记录成绩
        plot_scores.append(score)
        mean_score = np.mean(plot_scores[-50:])
        plot_mean_scores.append(mean_score)

        if score > record:
            record = score
            agent1.model.save("best_model_1.pth")
            agent2.model.save("best_model_2.pth")

        print(f"Game {agent1.n_game} | Score {score} | Record {record} | Mean {mean_score:.2f}")

        # 画图
        plot(plot_scores, plot_mean_scores)

        # UI 模式 B：每 100 局渲染一局，用当前训练好的模型玩一盘
        if agent1.n_game % 100 == 0:
            print("Rendering one episode with current policy...")
            game.render_one_episode(agent1, agent2)


if __name__ == "__main__":
    train()
