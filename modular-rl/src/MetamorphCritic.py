from copy import deepcopy

import torch
import torch.nn as nn

from MetamorphTransformer import MetaMorphTransformer, DEVICE
from torch.nn import functional as F


def _get_vals(vals, mask, num_limbs):
    vals[mask] = 0
    vals = torch.sum(vals, dim=1, keepdim=True)
    vals = torch.div(vals, num_limbs)
    return vals


class MetamorphCritic(nn.Module):
    def __init__(
            self,
            state_dim,
            action_dim,
            msg_dim,
            batch_size,
            max_children,
            disable_fold,
            td,
            bu,
            args=None,
    ):
        super(MetamorphCritic, self).__init__()
        self.seq_len = args.max_num_limbs  # Constant number same as max_children????
        self.num_limbs = 1
        self.batch_size = batch_size
        self.obs_padding_mask = self.calculate_padding_mask()
        self.max_children = max_children
        self.obs_dim = state_dim
        self.critic_1 = MetaMorphTransformer(state_dim + action_dim, 1, self.seq_len,
                                             args.attention_embedding_size,
                                             args.attention_heads,
                                             args.attention_hidden_size,
                                             args.attention_layers,
                                             args.dropout_rate)
        self.critic_2 = deepcopy(self.critic_1)
        self.obs_dim = state_dim
        self.action_dim = action_dim

    def __preprocess_input(self, obs, action):
        obs = obs.reshape(self.batch_size, -1, self.obs_dim)
        action = action.reshape(self.batch_size, -1, self.action_dim)
        _, num_limbs, _ = obs.shape
        pad_size = self.seq_len - num_limbs
        obs = F.pad(obs, [0, 0, 0, pad_size])
        action = F.pad(action, [0, 0, 0, pad_size])
        mask = [False] * num_limbs + [True] * pad_size
        mask = torch.tensor(mask, device=DEVICE)
        mask = mask.repeat(self.batch_size, 1)

        obs_act = torch.cat([obs, action], dim=-1)
        return obs_act, mask

    def forward(self, obs, action):
        """Calculate forward pass through both critics."""
        obs_act, mask = self.__preprocess_input(obs, action)
        val_1 = self.critic_1(obs_act, mask)
        val_2 = self.critic_2(obs_act, mask)
        val_1 = _get_vals(val_1, mask, self.num_limbs)
        val_2 = _get_vals(val_2, mask, self.num_limbs)
        return val_1, val_2

    def Q1(self, obs, action):
        """Calculate output from the first critic."""
        obs_act, mask = self.__preprocess_input(obs, action)
        value = self.critic_1(obs_act, mask)
        value = _get_vals(value, mask, self.num_limbs)
        return value

    def change_morphology(self, parents):
        self.num_limbs = len(parents)
        self.obs_padding_mask = self.calculate_padding_mask()

    def calculate_padding_mask(self):
        pad_size = self.seq_len - self.num_limbs
        obs_padding_mask = [False] * self.num_limbs + [True] * pad_size
        obs_padding_mask = torch.tensor(obs_padding_mask, device=DEVICE)
        return obs_padding_mask.repeat(self.batch_size, 1)
