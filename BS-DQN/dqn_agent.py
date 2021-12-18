import numpy as np
import torch as T
from deep_q_network import DeepQNetwork
from ensemble_q_network import EnsembleNet
from replay_memory import ReplayBuffer

class DQNAgent(object):
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-7, n_ensemble=3,
                 replace=1000, algo=None, env_name=None, chkpt_dir='tmp/bs-dqn'):

        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0

        self.n_ensemble = n_ensemble

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

        self.q_eval = EnsembleNet(chkpt_dir=chkpt_dir, name=self.env_name + '_' + self.algo + '_q_eval', n_ensemble=self.n_ensemble, n_actions=self.n_actions, lr=self.lr, input_dims=self.input_dims)
        self.q_next = EnsembleNet(chkpt_dir=chkpt_dir, name=self.env_name + '_' + self.algo + '_q_next', n_ensemble=self.n_ensemble, n_actions=self.n_actions, lr=self.lr, input_dims=self.input_dims)

       

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
            evals = self.q_eval.forward(state, None)
            values = []
            actions = []
            for head in range(self.n_ensemble):
                values.append(evals[head].max())
                actions.append(evals[head].argmax())

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

        total_loss = []
        sample_selections = T.FloatTensor(np.random.randint(2, size=(self.n_ensemble, self.batch_size)))
        for head in range(self.n_ensemble):
            q_preds = self.q_eval.forward(states, head)
            q_nexts = self.q_next.forward(states_, head)
            q_nexts[dones] = 0.0

            q_targets = rewards + self.gamma * q_nexts.max(dim=1).values * sample_selections[head]
            q_preds = q_preds.max(dim=1).values * sample_selections[head] 
            total_loss.append(self.q_eval.loss(q_targets, q_preds).to(self.q_eval.device))

        loss = sum(total_loss) / self.n_ensemble
        loss.backward()

        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()
