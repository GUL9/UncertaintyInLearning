import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_results_from_dir(path_to_dir):
    results = []
    for filename in sorted(os.listdir(path_to_dir)):
        if filename != 'not-used':
            print(f'READING RESULTS FROM: {filename}')
            results.append(pd.read_csv(path_to_dir + filename))

    return results

def plot_dropout_scores(dropout_results):
    dropout_scores = pd.merge(dropout_results[0], dropout_results[1],on='episode', suffixes=("_p005", "_p01"))
    dropout_scores = pd.merge(dropout_scores, dropout_results[2], on='episode').rename(columns={'score': 'score_p015', 'epsilon': 'epsilon_p015', 'budget': 'bugdet_p015', 'uncertainty': 'uncertainty_p015'})
    dropout_scores.plot(x='episode', y=['score_p005', 'score_p01', 'score_p015'])
    plt.show()

def score_to_avg(df, window_size):
    sumbuff = np.zeros(window_size)
    for index, row in df.iterrows():
        sumbuff[index % window_size] = row['score']
        row['score'] = np.sum(sumbuff) / (index+1 if index+1 < window_size else window_size)
        df.iloc[index] = row

    return df

def plot_dropout_scores_avg(dropout_results, window_size):
    dropout_results = [score_to_avg(df, window_size) for df in dropout_results]
            
    dropout_scores = pd.merge(dropout_results[0], dropout_results[1],on='episode', suffixes=(f"_avg{window_size}_p005", f"_avg{window_size}_p01"))
    dropout_scores = pd.merge(dropout_scores, dropout_results[2], on='episode').rename(columns={'score': f'score_avg{window_size}_p015', 'epsilon': 'epsilon_p015', 'budget': 'bugdet_p015', 'uncertainty': 'uncertainty_p015'})
    dropout_scores.plot(x='episode', y=[f'score_avg{window_size}_p005', f'score_avg{window_size}_p01', f'score_avg{window_size}_p015'])
    plt.show()

def plot_bs_scores(bs_results,):
    bs_scores = pd.merge(bs_results[0], bs_results[1],on='episode', suffixes=("_pscale10", "_pscale11"))
    bs_scores = pd.merge(bs_scores, bs_results[2], on='episode').rename(columns={'score': 'score_pscale12', 'epsilon': 'epsilon_pscale12', 'budget': 'bugdet_pscale12', 'uncertainty': 'uncertainty_pscale12'})
    bs_scores.autocorrelation_plot(x='episode', y=['score_pscale10', 'score_pscale11', 'score_pscale12'])
    plt.show()

def plot_bs_scores_avg(bs_results, window_size):
    bs_results = [score_to_avg(df, window_size) for df in bs_results]
            
    bs_scores = pd.merge(bs_results[0], bs_results[1],on='episode', suffixes=(f"_avg{window_size}_pscale10", f"_avg{window_size}_pscale11"))
    bs_scores = pd.merge(bs_scores, bs_results[2], on='episode').rename(columns={'score': f'score_avg{window_size}_pscale12', 'epsilon': 'epsilon_pscale12', 'budget': 'bugdet_pscale12', 'uncertainty': 'uncertainty_pscale12'})
    bs_scores.plot(x='episode', y=[f'score_avg{window_size}_pscale10', f'score_avg{window_size}_pscale11', f'score_avg{window_size}_pscale12'])
    plt.show()




    

results_path = "/tmp/scores/"
dqn_path = "DQN" + results_path
dropout_path = "Dropout-DQN" + results_path
bs_path = "BS-DQN" + results_path
bsd_path = "BS-Dropout-DQN" + results_path


dqn_results = read_results_from_dir(dqn_path)
dropout_results = read_results_from_dir(dropout_path)
bs_results = read_results_from_dir(bs_path)
bsd_results = read_results_from_dir(bsd_path)

avg_window_size = 30

#plot_dropout_scores(dropout_results)
#plot_dropout_scores_avg(dropout_results, avg_window_size)

#plot_bs_scores(bs_results)
#plot_bs_scores_avg(bs_results, avg_window_size)


