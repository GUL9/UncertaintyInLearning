import gym
import numpy as np
from dqn_agent import DQNAgent
from utils import plot_learning_curve, make_env, save_scores_csv
from gym import wrappers

if __name__ == '__main__':
    env = make_env('PongNoFrameskip-v4')
    #env = gym.make('CartPole-v1')
    best_score = -np.inf
    load_checkpoint = False
    n_games = 300

    agent = DQNAgent(gamma=0.99, epsilon=1, lr=0.0001,
                     input_dims=(env.observation_space.shape),
                     n_actions=env.action_space.n, mem_size=50000, eps_min=0.05,
                     batch_size=32, replace=1000, eps_dec=1e-5,
                     chkpt_dir='models/', algo='DropoutAgentP01UQ01',
                     env_name='PongNoFrameskip-v4')

    if load_checkpoint:
        agent.load_models()

    fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) +'_' + str(n_games) + 'games'
    figure_file = 'tmp/plots/' + fname + '.png'
    score_file = 'tmp/scores/' + fname + '.csv'
    # if you want to record video of your agent playing, do a mkdir tmp && mkdir tmp/dqn-video
    # and uncomment the following 2 lines.
    env = wrappers.Monitor(env, "tmp/dqn-video", video_callable=lambda episode_id: True, force=True)
    n_steps = 0
    scores, eps_history, steps_array, budget, uncertainties = [], [], [], [], []

    for i in range(n_games):
        done = False
        observation = env.reset()

        score = 0
        game_uncertainty = []
        while not done:
            action, uncertainty = agent.choose_action(observation)
            game_uncertainty.append(uncertainty)
            env.render()
            observation_, reward, done, info = env.step(action)
            score += reward

            if not load_checkpoint:
                agent.store_transition(observation, action, reward, observation_, done)
                agent.learn()
            observation = observation_
            n_steps += 1
        uncertainties.append(np.mean(game_uncertainty))
        budget.append(agent.advice_budget)
        scores.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])
        print('episode: ', i,
            'score: ', score,
            'average score: %.1f' % avg_score, 
            'best score: %.2f' % best_score,
            'epsilon: %.2f' % agent.epsilon,
            'steps:', n_steps,
            'budget:', budget[-1],
            'uncertainty: %.4f ' % uncertainties[-1])

        if avg_score > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = avg_score

        eps_history.append(agent.epsilon)


    save_scores_csv(scores, eps_history, score_file, budget=budget, uncertainty=uncertainties)
    plot_learning_curve(steps_array, scores, eps_history, figure_file)
