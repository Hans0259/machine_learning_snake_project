import numpy as np
from snake_env_CNN import Game
from snake_agent_CNN import Agent
from helper import plot 

def train():
    plot_scores = []
    plot_mean = []
    record = 0

    game = Game(width=320, height=320)
    agent = Agent(game.nC, game.nA, game.grid_h, game.grid_w)

    total_steps = 0

    while True:
        # start
        game.reset()
        state = game.get_grid_obs()
        done = False
        score = 0

        while not done:
            action = agent.get_action(state, explore=True)
            reward, done, score = game.play_step(action)
            next_state = game.get_grid_obs()

            agent.remember(state, action, reward, next_state, done)
            agent.train_long_memory(batch_size=128)

            state = next_state
            total_steps += 1

            # update the target network
            if total_steps % 1000 == 0:
                agent.update_target()

        # end of a game
        agent.n_game += 1

        plot_scores.append(score)
        mean_score = np.mean(plot_scores[-50:])
        plot_mean.append(mean_score)

        if score > record:
            record = score
            agent.save("best_model.pth")

        print(
            f"Game {agent.n_game} | Score {score} | "
            f"Record {record} | Mean(50) {mean_score:.2f}"
        )

        plot(plot_scores, plot_mean)

        # see result for every 50 games
        if agent.n_game % 50 == 0:
            print("Rendering one episode with current policy...")
            game.render_one_episode(agent)


if __name__ == "__main__":
    train()
