import numpy as np
import torch as T
from deep_q_network import DropoutQNetwork
from deep_q_network import DeepQNetwork
from ensemble_q_network import EnsembleNet
from replay_memory import ReplayBuffer

class DQNAgent(object):
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-7, n_ensemble=5,
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

        self.n_ensemble = n_ensemble
        self.n_actions = n_actions
        self.action_space = [i for i in range(n_actions)]

        self.input_dims = input_dims
        self.batch_size = batch_size
        self.replace_target_cnt = replace
        self.learn_step_counter = 0
        self.advice_budget = 10000

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

        self.q_eval = DropoutQNetwork(lr=self.lr, n_actions=self.n_actions, name=self.env_name+'_'+self.algo+'_q_eval', input_dims=self.input_dims, chkpt_dir=self.chkpt_dir)
        self.q_next = DropoutQNetwork(lr=self.lr, n_actions=self.n_actions, name=self.env_name+'_'+self.algo+'_q_eval', input_dims=self.input_dims, chkpt_dir=self.chkpt_dir)
        self.q_advice = DeepQNetwork(lr=self.lr, n_actions=self.n_actions, name='PongNoFrameskip-v4_DQNAgent_q_eval', input_dims=self.input_dims, chkpt_dir=self.advice_dir)
        self.q_advice.load_checkpoint()
    

    def choose_action(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
        evals, values, actions = [], [], []
        for _ in range(10):
            eval = self.q_eval.forward(state)
            evals.append(eval)
            values.append(T.max(eval))
            actions.append(T.argmax(eval))

        uncertainty = T.var(T.tensor(values))
        if uncertainty < 0.001 or self.advice_budget <= 0:
            action = self._std_policy(values, actions)
        else:
            print(uncertainty)
            action = self._advice_policy(state)
        return action, uncertainty

    def _advice_policy(self, state):
        advice = self.q_advice.forward(state)
        action = T.argmax(advice).item()

        self.advice_budget -= 1
        
        return action

    def _std_policy(self, values, actions):
        if np.random.random() > self.epsilon:
            best_head = T.tensor(values).argmax()
            action = actions[best_head]
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
        indices = np.arange(self.batch_size)

        q_pred = self.q_eval.forward(states)[indices, actions]
        q_next = self.q_next.forward(states_).max(dim=1)[0]

        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next

        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()