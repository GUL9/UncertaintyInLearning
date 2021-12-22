import numpy as np
import torch as T
from deep_q_network import DeepQNetwork
from ensemble_q_network import EnsembleNet, EnsembleWithPrior
from replay_memory import ReplayBuffer

class BSDropoutAgent(object):
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims, mem_size, batch_size, 
                eps_min=0.01, eps_dec=5e-7, n_ensemble=5, replace=1000, 
                algo=None, env_name=None, chkpt_dir='models/'):

        self.env_name = env_name
        self.algo = algo
        self.chkpt_dir = chkpt_dir
        self.advice_dir = chkpt_dir + 'advice_model'

        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.lr = lr

        self.n_ensemble = n_ensemble
        self.n_actions = n_actions
        self.action_space = [i for i in range(n_actions)]

        self.input_dims = input_dims
        self.batch_size = batch_size
        self.replace_target_cnt = replace
        self.learn_step_counter = 0
        self.advice_budget = 50000

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

        self.e_prior = EnsembleNet(chkpt_dir=chkpt_dir, name=self.env_name + '_' + self.algo + '_e_prior', n_ensemble=self.n_ensemble, n_actions=self.n_actions, lr=self.lr, input_dims=self.input_dims)
        self.e_eval = EnsembleNet(chkpt_dir=chkpt_dir, name=self.env_name + '_' + self.algo + '_e_eval', n_ensemble=self.n_ensemble, n_actions=self.n_actions, lr=self.lr, input_dims=self.input_dims)
        self.e_next = EnsembleNet(chkpt_dir=chkpt_dir, name=self.env_name + '_' + self.algo + '_e_next', n_ensemble=self.n_ensemble, n_actions=self.n_actions, lr=self.lr, input_dims=self.input_dims)

        self.e_prior.init_heads()
        self.e_eval = EnsembleWithPrior(chkpt_dir + self.env_name + '_' + self.algo + '_q_eval', self.e_eval, self.e_prior, prior_scale=11, lr=lr)
        self.e_next = EnsembleWithPrior(chkpt_dir + self.env_name + '_' + self.algo + '_q_next', self.e_next, self.e_prior, prior_scale=11, lr=lr)

        self.a_eval = DeepQNetwork(lr=self.lr, n_actions=self.n_actions, name= self.env_name + '_' + self.algo + '_a_eval', input_dims=self.input_dims, chkpt_dir=chkpt_dir)
        self.a_next = DeepQNetwork(lr=self.lr, n_actions=self.n_actions, name= self.env_name + '_' + self.algo + '_a_next', input_dims=self.input_dims, chkpt_dir=chkpt_dir)

        self.q_advice = DeepQNetwork(lr=self.lr, n_actions=self.n_actions, name='PongNoFrameskip-v4_DQNAgent_q_eval', input_dims=self.input_dims, chkpt_dir=self.advice_dir)
        self.q_advice.load_checkpoint()
    

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.e_eval.device)
        
        e_evals, e_uncertainty = self._epistemic_uncertainty(state)
        a_evals, a_uncertainty = self._aleatoric_uncertainty(state)
        
        uncertainty = e_uncertainty + a_uncertainty

        if uncertainty < 0.2 or self.advice_budget <= 0:
            action = self._std_policy(e_evals, a_evals)
        else:
            action = self._advice_policy(state)
        return action, uncertainty

    def _aleatoric_uncertainty(self, state):
        evals = T.tensor([]).to(self.a_eval.device)
        for _ in range(10):
            evals = T.cat((evals, self.a_eval.forward(state, drop=True)), 0)
        
        action_means = T.mean(evals, dim=0)
        best_action = T.argmax(action_means).item()
        uncertainty = T.var(evals, dim=0)[best_action].item()
        return evals, uncertainty

    def _epistemic_uncertainty(self, state):
        evals = self.e_eval.forward(state, None)
        evals_ = T.tensor([])
        for head_eval in evals:
            evals_ = T.cat((evals_, head_eval), dim=0)
        evals = evals_
        
        action_means = T.mean(evals, dim=0)
        best_action = T.argmax(action_means).item()
        uncertainty = T.var(evals, dim=0)[best_action].item()
        return evals, uncertainty

    def _advice_policy(self, state):
        advice = self.q_advice.forward(state)
        action = T.argmax(advice).item()

        self.advice_budget -= 1
        
        return action

    def _std_policy(self, e_evals, a_evals):
        if np.random.random() > self.epsilon:
            evals = T.cat((e_evals, a_evals), dim=0)
            action_means = T.mean(evals, dim=0)
            action = T.argmax(action_means).item()
        else: 
            action = np.random.choice(self.action_space)
            
        return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state).to(self.e_eval.device)
        rewards = T.tensor(reward).to(self.e_eval.device)
        dones = T.tensor(done).to(self.e_eval.device)
        actions = T.tensor(action).to(self.e_eval.device)
        states_ = T.tensor(new_state).to(self.e_eval.device)

        return states, actions, rewards, states_, dones

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.e_next.load_state_dict(self.e_eval.state_dict())
            self.a_next.load_state_dict(self.a_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def save_models(self):
        self.e_eval.save_checkpoint()
        self.e_next.save_checkpoint()
        self.a_eval.save_checkpoint()
        self.a_next.save_checkpoint()

    def load_models(self):
        self.e_eval.load_checkpoint()
        self.e_next.load_checkpoint()
        self.a_eval.load_checkpoint()
        self.a_next.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.replace_target_network()
        states, actions, rewards, states_, dones = self.sample_memory()


        self._epistemic_learn(states, actions, rewards, states_, dones)
        self._aleatoric_learn(states, actions, rewards, states_, dones)

        self.learn_step_counter += 1
        self.decrement_epsilon()

    def _aleatoric_learn(self, states, actions, rewards, states_, dones):
        self.a_eval.optimizer.zero_grad()
        indices = np.arange(self.batch_size)

        q_pred = self.a_eval.forward(states, drop=False)[indices, actions]
        q_next = self.a_next.forward(states_, drop=False).max(dim=1)[0]

        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next
        
        loss = self.a_eval.loss(q_target, q_pred).to(self.a_eval.device)
        loss.backward()
        self.a_eval.optimizer.step()

    def _epistemic_learn(self, states, actions, rewards, states_, dones):
        self.e_eval.optimizer.zero_grad()
        total_loss = []
        sample_selections = T.FloatTensor(np.random.randint(2, size=(self.n_ensemble, self.batch_size))).to(self.e_eval.device)
        for head in range(self.n_ensemble):
            indices = np.arange(self.batch_size)
            q_preds = self.e_eval.forward(states, head)[indices, actions]
            q_nexts = self.e_next.forward(states_, head).max(dim=1).values
            q_nexts[dones] = 0.0
            
            q_targets = rewards + self.gamma * q_nexts * sample_selections[head]
            q_preds = q_preds * sample_selections[head] 
            total_loss.append(self.e_eval.loss(q_targets, q_preds).to(self.e_eval.device))

        loss = sum(total_loss) / self.n_ensemble
        loss.backward()

        self.e_eval.optimizer.step()
