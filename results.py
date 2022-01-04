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

def score_to_avg(df, window_size):
    sumbuff = np.zeros(window_size)
    for index, row in df.iterrows():
        sumbuff[index % window_size] = row['score']
        row['score'] = np.sum(sumbuff) / (index+1 if index+1 < window_size else window_size)
        df.iloc[index] = row

    return df

def score_to_cumsum(df, window_size):
    sumbuff = np.zeros(window_size)
    for index, row in df.iterrows():
        sumbuff[index % window_size] = row['score']
        row['score'] = np.sum(sumbuff)
        df.iloc[index] = row

    return df


def plot_dropout_scores(dropout_results):
    dropout_scores = pd.merge(dropout_results[0], dropout_results[1],on='episode', suffixes=("_p005", "_p01"))
    dropout_scores = pd.merge(dropout_scores, dropout_results[2], on='episode').rename(columns={'score': 'score_p015'})
    dropout_scores.plot(x='episode', y=['score_p005', 'score_p01', 'score_p015'])
    plt.show()

def plot_dropout_scores_avg(dropout_results, window_size):
    dropout_results = [score_to_avg(df, window_size) for df in dropout_results]
            
    dropout_scores = pd.merge(dropout_results[0], dropout_results[1],on='episode', suffixes=(f"_avg{window_size}_p005", f"_avg{window_size}_p01"))
    dropout_scores = pd.merge(dropout_scores, dropout_results[2], on='episode').rename(columns={'score': f'score_avg{window_size}_p015'})
    dropout_scores.plot(x='episode', y=[f'score_avg{window_size}_p005', f'score_avg{window_size}_p01', f'score_avg{window_size}_p015'])
    plt.show()

def plot_dropout_cumsum(dropout_results, window_size):
    dropout_results = [score_to_cumsum(df, window_size) for df in dropout_results]
            
    dropout_scores = pd.merge(dropout_results[0], dropout_results[1],on='episode', suffixes=(f"_sum{window_size}_p005", f"_sum{window_size}_p01"))
    dropout_scores = pd.merge(dropout_scores, dropout_results[2], on='episode').rename(columns={'score': f'score_sum{window_size}_p015'})
    dropout_scores.plot(x='episode', y=[f'score_sum{window_size}_p005', f'score_sum{window_size}_p01', f'score_sum{window_size}_p015'])
    plt.show()

def plot_bs_scores(bs_results,):
    bs_scores = pd.merge(bs_results[0], bs_results[1],on='episode', suffixes=("_pscale10", "_pscale11"))
    bs_scores = pd.merge(bs_scores, bs_results[2], on='episode').rename(columns={'score': 'score_pscale12'})
    bs_scores.plot(x='episode', y=['score_pscale10', 'score_pscale11', 'score_pscale12'])
    plt.show()

def plot_bs_scores_avg(bs_results, window_size):
    bs_results = [score_to_avg(df, window_size) for df in bs_results]
            
    bs_scores = pd.merge(bs_results[0], bs_results[1],on='episode', suffixes=(f"_avg{window_size}_pscale10", f"_avg{window_size}_pscale11"))
    bs_scores = pd.merge(bs_scores, bs_results[2], on='episode').rename(columns={'score': f'score_avg{window_size}_pscale12'})
    bs_scores.plot(x='episode', y=[f'score_avg{window_size}_pscale10', f'score_avg{window_size}_pscale11', f'score_avg{window_size}_pscale12'])
    plt.show()

def plot_bs_cumsum(bs_results, window_size):
    bs_results = [score_to_avg(df, window_size) for df in bs_results]
            
    bs_scores = pd.merge(bs_results[0], bs_results[1],on='episode', suffixes=(f"_sum{window_size}_pscale10", f"_sum{window_size}_pscale11"))
    bs_scores = pd.merge(bs_scores, bs_results[2], on='episode').rename(columns={'score': f'score_sum{window_size}_pscale12'})
    bs_scores.plot(x='episode', y=[f'score_sum{window_size}_pscale10', f'score_sum{window_size}_pscale11', f'score_sum{window_size}_pscale12'])
    plt.show()


def plot_scores_compare_all(dqn_results, dropout_results, bs_results, bsd_results):
    scores1 = pd.merge(dqn_results[0], bsd_results[0], on='episode', suffixes=('_DQN', '_bsd_p01_pscale11'))
    scores2 = pd.merge(dropout_results[1], bs_results[1], on='episode', suffixes=('_dropout_p01', '_bs_pscale11'))
    scores = pd.merge(scores1, scores2, on='episode')
    scores.plot(x='episode', y=['score_DQN', 'score_bsd_p01_pscale11', 'score_dropout_p01', 'score_bs_pscale11'])
    plt.show()

def plot_scores_avg_compare_all(dqn_results, dropout_results, bs_results, bsd_results, window_size):
    avg_scores = [score_to_avg(dqn_results[0], window_size), score_to_avg(bsd_results[0], window_size), score_to_avg(dropout_results[1], window_size), score_to_avg(bs_results[1],window_size)]

    scores1 = pd.merge(avg_scores[0], avg_scores[1], on='episode', suffixes=('_avg_DQN', '_avg_bsd_p01_pscale11'))
    scores2 = pd.merge(avg_scores[2], avg_scores[3], on='episode', suffixes=('_avg_dropout_p01', '_avg_bs_pscale11'))
    scores = pd.merge(scores1, scores2, on='episode')
    scores.plot(x='episode', y=['score_avg_DQN', 'score_avg_bsd_p01_pscale11', 'score_avg_dropout_p01', 'score_avg_bs_pscale11'])
    plt.show()

def plot_scores_cumsum_compare_all(dqn_results, dropout_results, bs_results, bsd_results, window_size):
    cumsum_scores = [score_to_cumsum(dqn_results[0], window_size), score_to_cumsum(bsd_results[0], window_size), score_to_cumsum(dropout_results[1], window_size), score_to_cumsum(bs_results[1],window_size)]

    scores1 = pd.merge(cumsum_scores[0], cumsum_scores[1], on='episode', suffixes=('_cumsum_DQN', '_cumsum_bsd_p01_pscale11'))
    scores2 = pd.merge(cumsum_scores[2], cumsum_scores[3], on='episode', suffixes=('_cumsum_dropout_p01', '_cumsum_bs_pscale11'))
    scores = pd.merge(scores1, scores2, on='episode')
    scores.plot(x='episode', y=['score_cumsum_DQN', 'score_cumsum_bsd_p01_pscale11', 'score_cumsum_dropout_p01', 'score_cumsum_bs_pscale11'])
    plt.show()

def plot_uncertainties_all(dropout_results, bs_results, bsd_results):
    uncertainties = pd.merge(dropout_results[1], bs_results[1], on='episode', suffixes=('_dropout_p01 (aleatoric)', '_bs_pscale11 (epistemic)'))
    uncertainties = pd.merge(uncertainties, bsd_results[0],  on='episode').rename(columns={'uncertainty': 'uncertainty_bsd_p01_pscale11 (total)', 'aleatoric': 'uncertainty_bsd_p01_pscale11 (aleatoric)', 'epistemic': 'uncertainty_bsd_p01_pscale11 (epistemic)'})
    uncertainties.plot(x='episode', y=['uncertainty_bsd_p01_pscale11 (total)', 'uncertainty_bsd_p01_pscale11 (epistemic)', 'uncertainty_bsd_p01_pscale11 (aleatoric)', 'uncertainty_dropout_p01 (aleatoric)', 'uncertainty_bs_pscale11 (epistemic)'])
    plt.show()


def plot_budgets_all(dropout_results, bs_results, bsd_results):
    uncertainties = pd.merge(dropout_results[1], bs_results[1], on='episode', suffixes=('_dropout_p01', '_bs_pscale11'))
    uncertainties = pd.merge(uncertainties, bsd_results[0],  on='episode').rename(columns={'budget': 'budget_bsd_p01_pscale11'})
    uncertainties.plot(x='episode', y=['budget_bsd_p01_pscale11', 'budget_dropout_p01', 'budget_bs_pscale11'])
    plt.show()

def plot_dropout_budgets(dropout_results):
    dropout_budgets = pd.merge(dropout_results[0], dropout_results[1],on='episode', suffixes=("_p005", "_p01"))
    dropout_budgets = pd.merge(dropout_budgets, dropout_results[2], on='episode').rename(columns={'budget': 'budget_p015'})
    dropout_budgets.plot(x='episode', y=['budget_p005', 'budget_p01', 'budget_p015'])
    plt.show()

def plot_bs_budgets(bs_results,):
    bs_scores = pd.merge(bs_results[0], bs_results[1],on='episode', suffixes=("_pscale10", "_pscale11"))
    bs_scores = pd.merge(bs_scores, bs_results[2], on='episode').rename(columns={'budget': 'budget_pscale12'})
    bs_scores.plot(x='episode', y=['budget_pscale10', 'budget_pscale11', 'budget_pscale12'])
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

avg_window_size = 300

#plot_dropout_scores(dropout_results)
#plot_dropout_scores_avg(dropout_results, avg_window_size)
plot_dropout_cumsum(dropout_results, avg_window_size)

#plot_bs_scores(bs_results)
#plot_bs_scores_avg(bs_results, avg_window_size)
plot_bs_cumsum(bs_results, avg_window_size)


#plot_scores_compare_all(dqn_results, dropout_results, bs_results, bsd_results)
#plot_scores_avg_compare_all(dqn_results, dropout_results, bs_results, bsd_results, avg_window_size)
plot_scores_cumsum_compare_all(dqn_results, dropout_results, bs_results, bsd_results, avg_window_size)

#plot_uncertainties_all(dropout_results, bs_results, bsd_results)

#plot_dropout_budgets(dropout_results)
#plot_bs_budgets(bs_results)
#plot_budgets_all(dropout_results, bs_results, bsd_results)




