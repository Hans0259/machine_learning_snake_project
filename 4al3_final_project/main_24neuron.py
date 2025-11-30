import numpy as np
from snake_env_24neuron import Game
from snake_agent_24neuron import Agent
from helper import plot


def train():
    plot_scores = []
    plot_mean_scores = []
    record = 0
    total_step = 0

    game = Game()
    agent1 = Agent(game.nS, game.nA)
    agent2 = Agent(game.nS, game.nA)

    state1_new = game.get_state(state=1)
    state2_new = game.get_state(state=2)

    while True:
        state1_old = state1_new
        state2_old = state2_new

        action1 = agent1.get_action(state1_old, agent1.n_game)
        action2 = agent2.get_action(state2_old, agent2.n_game)

        reward, done, score = game.play_step(action1, action2)

        state1_new = game.get_state(state=1)
        state2_new = game.get_state(state=2)

        agent1.remember(state1_old, action1, reward, state1_new, done)
        agent2.remember(state2_old, action2, reward, state2_new, done)

        agent1.train_long_memory(batch_size=256)
        agent2.train_long_memory(batch_size=256)

        total_step += 1

        if total_step % 10 == 0:
            agent1.trainer.copy_model()
            agent2.trainer.copy_model()

        if done:
            game.reset()
            agent1.n_game += 1
            agent2.n_game += 1

            if score > record:
                record = score
                agent1.trainer.model.save()
                agent2.trainer.model.save()

            print('Game', agent1.n_game, 'Score', score, 'Record:', record)
            plot_scores.append(score)
            mean_score = np.mean(plot_scores[-10:])
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

if __name__ == '__main__':
    train()



