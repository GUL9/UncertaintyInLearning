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
    elif classtype == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    else:
        print('%s is not initialized.' %classtype)


class CoreNet(nn.Module):
    def __init__(self, input_dims):
        super(CoreNet, self).__init__()
        # params from ddqn appendix
        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride = 4)
        # TODO - should we have this init during PRIOR code?
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
        self.fc1.apply(weights_init)
        self.fc2.apply(weights_init)

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
        self.net_list = nn.ModuleList([HeadNet(n_actions=n_actions) for head in range(n_ensemble)])

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()



        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def _core(self, x):
        return self.core_net(x)

    def _heads(self, x):
        preds = T.tensor([]).to(self.device)
        for head in self.net_list:
            preds = T.cat((preds, head(x)), 0)
        return preds

    def forward(self, x, head):
        if head is None:
            core_cache = self._core(x)
            net_heads = self._heads(core_cache)
            return net_heads
        else:
            return self.net_list[head](self.core_net(x))


    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        print(self.checkpoint_file)
        self.load_state_dict(T.load(self.checkpoint_file))