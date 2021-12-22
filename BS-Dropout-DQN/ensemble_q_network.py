# Model style from Kyle @
# https://gist.github.com/kastnerkyle/a4498fdf431a3a6d551bcc30cd9a35a0
import torch.nn as nn
import torch.nn.functional as F
import torch as T
import torch.optim as optim
import numpy as np
import os


# from the DQN paper
#The first convolution layer convolves the input with 32 filters of size 8 (stride 4),
#the second layer has 64 layers of size 4
#(stride 2), the final convolution layer has 64 filters of size 3 (stride
#1). This is followed by a fully-connected hidden layer of 512 units.

# init func used by hengyaun
def weights_init(m):
    """custom weights initialization"""
    classtype = m.__class__
    if classtype == nn.Linear or classtype == nn.Conv2d:
        print("default init")
        T.nn.init.xavier_normal(m.weight)
        T.nn.init.zeros_(m.bias)
    else:
        print('%s is not initialized.' %classtype)


class CoreNet(nn.Module):
    def __init__(self, input_dims):
        super(CoreNet, self).__init__()
        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride = 4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # size after conv3
        reshape = 64*7*7
        x = x.view(-1, reshape)
        return x

class HeadNet(nn.Module):
    def __init__(self, n_actions):
        super(HeadNet, self).__init__()
        mult = 64*7*7
        self.fc1 = nn.Linear(mult, 512)
        self.fc2 = nn.Linear(512, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x   

class EnsembleNet(nn.Module):
    def __init__(self, chkpt_dir, name, n_ensemble, n_actions, lr, input_dims):
        super(EnsembleNet, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.core_net = CoreNet(input_dims=input_dims)
        self.head_nets = nn.ModuleList([HeadNet(n_actions=n_actions) for head in range(n_ensemble)])

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def _core(self, x):
        return self.core_net(x)

    def _heads(self, x):
        return [head(x) for head in self.head_nets]

    def init_heads(self):
        for head in self.head_nets:
            head.apply(weights_init)

    def forward(self, x, head):
        if head is None:
            core_cache = self._core(x)
            net_heads = self._heads(core_cache)
            return net_heads
        else:
            return self.head_nets[head](self.core_net(x))


    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))


class EnsembleWithPrior(nn.Module):
    def __init__(self, checkpoint_file, net, prior_net, prior_scale=1., lr=0.001):
        super(EnsembleWithPrior, self).__init__()
        self.checkpoint_file = checkpoint_file

        self.net = net

        self.prior_scale = prior_scale
        self.prior = prior_net if self.prior_scale > 0. else None

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def _forward_all_heads(self, x):
        core_output = self.net._core(x)
        head_outputs = self.net._heads(core_output)

        if self.prior_scale <= 0.:
            return head_outputs
        else:
            prior_core_output = self.prior._core(x)
            prior_head_outputs = self.prior._heads(prior_core_output)
            preds = []
            for head_o, prior_o in zip(head_outputs, prior_head_outputs):
                pred = head_o + self.prior_scale * prior_o.detach()
                preds.append(pred)
            return preds
    
    def _forward_single_head(self, x, head):

        if self.prior_scale > 0.:
            return self.net(x, head) + self.prior_scale * self.prior(x, head).detach()
        else:
            return self.net(x, head)

    def forward(self, x, head):
        if hasattr(self.net, "head_nets"):
            return self._forward_all_heads(x) if head is None else  self._forward_single_head(x, head) 
        else:
            raise ValueError("Ensemble missing head nets. Must have attribute: head_nets")

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))