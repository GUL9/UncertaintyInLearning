import numpy as np
import torch as T
from deep_q_network import DeepQNetwork
from ensemble_q_network import EnsembleNet, EnsembleWithPrior
from replay_memory import ReplayBuffer

class DQNAgent(object):
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, uncertainty_threshold=0.1, eps_min=0.01, eps_dec=5e-7, n_ensemble=5, prior_scale=1.0,
                 replace=1000, algo=None, env_name=None, chkpt_dir='models/'):

        self.env_name = env_name
        self.algo = algo
        self.chkpt_dir = chkpt_dir
        self.advice_dir = chkpt_dir + 'advice_model'

        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.lr = lr

        self.uncertainty_threshold = uncertainty_threshold
        self.prior_scale = prior_scale

        self.n_ensemble = n_ensemble
        self.n_actions = n_actions
        self.action_space = [i for i in range(n_actions)]

        self.input_dims = input_dims
        self.batch_size = batch_size
        self.replace_target_cnt = replace
        self.learn_step_counter = 0
        self.advice_budget = 50000

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

        self.q_prior = EnsembleNet(chkpt_dir=chkpt_dir, name=self.env_name + '_' + self.algo + '_q_prior', n_ensemble=self.n_ensemble, n_actions=self.n_actions, lr=self.lr, input_dims=self.input_dims)
        self.q_eval = EnsembleNet(chkpt_dir=chkpt_dir, name=self.env_name + '_' + self.algo + '_q_eval', n_ensemble=self.n_ensemble, n_actions=self.n_actions, lr=self.lr, input_dims=self.input_dims)
        self.q_next = EnsembleNet(chkpt_dir=chkpt_dir, name=self.env_name + '_' + self.algo + '_q_next', n_ensemble=self.n_ensemble, n_actions=self.n_actions, lr=self.lr, input_dims=self.input_dims)

        self.q_prior.init_heads()
        self.q_eval = EnsembleWithPrior(chkpt_dir + self.env_name + '_' + self.algo + '_q_eval', self.q_eval, self.q_prior, prior_scale=self.prior_scale, lr=lr)
        self.q_next = EnsembleWithPrior(chkpt_dir + self.env_name + '_' + self.algo + '_q_next', self.q_next, self.q_prior, prior_scale=self.prior_scale, lr=lr)

        self.q_advice = DeepQNetwork(lr=self.lr, n_actions=self.n_actions, name='PongNoFrameskip-v4_DQNAgent300Games_q_eval', input_dims=self.input_dims, chkpt_dir=self.advice_dir)
        self.q_advice.load_checkpoint()
    

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
        evals = self.q_eval.forward(state, None)
        a = T.tensor([]).to(self.q_eval.device)
        for head_eval in evals:
            a = T.cat((a, head_eval), dim=0)
        evals = a
        action_means = T.mean(evals, dim=0)
        best_action = T.argmax(action_means).item()
        uncertainty = T.var(evals, dim=0)[best_action].item()
        
        if uncertainty < self.uncertainty_threshold  or self.advice_budget <= 0:
            action = self._std_policy(best_action)
        else:
            action = self._advice_policy(state)
        return action, uncertainty

    def _advice_policy(self, state):
        advice = self.q_advice.forward(state)
        action = T.argmax(advice).item()

        self.advice_budget -= 1
        
        return action

    def _std_policy(self, best_action):
        if np.random.random() > self.epsilon:
            action = best_action
        else: 
            action = np.random.choice(self.action_space)
            
        return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        states_ = T.tensor(new_state).to(self.q_eval.device)

        return states, actions, rewards, states_, dones

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()


    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()
        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()

        total_loss = []
        sample_selections = T.FloatTensor(np.random.randint(2, size=(self.n_ensemble, self.batch_size))).to(self.q_eval.device)
        for head in range(self.n_ensemble):
            indices = np.arange(self.batch_size)
            q_preds = self.q_eval.forward(states, head)[indices, actions]
            q_nexts = self.q_next.forward(states_, head).max(dim=1).values
            q_nexts[dones] = 0.0
            
            q_targets = rewards + self.gamma * q_nexts * sample_selections[head]
            q_preds = q_preds * sample_selections[head] 
            total_loss.append(self.q_eval.loss(q_targets, q_preds).to(self.q_eval.device))

        loss = sum(total_loss) / self.n_ensemble
        loss.backward()

        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()
