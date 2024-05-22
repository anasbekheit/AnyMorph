import torch
from torch import nn
from MetamorphTransformer import MetaMorphTransformer, DEVICE
import torch.nn.functional as F


# TODO:
#  1) Deal with the init signature DONE
#  2) Remove reliability on config (know MAX LIMBS, ~DEVICE~) DONE
#  3) Check the num actions is 2 per model limb? DONE (1 per limb)
#  4) figure out how to deal with masking, as metamorph masks unused inputs (obs mask :True for unused, False for used)
#  4 cont: probably will calculate the padding mask, update it using change morphology
#  4 cont: obs_padding_mask = [False] * num_limbs + [True] * self.num_limb_pads ??
#  5) Add a change morphology method (will probably change the padding mask?)
class MetamorphActor(nn.Module):
    def __init__(
            self,
            state_dim,
            action_dim,
            msg_dim,
            batch_size,
            max_action,
            max_children,
            disable_fold,
            td,
            bu,
            args=None,
    ):
        super(MetamorphActor, self).__init__()
        self.seq_len = args.max_num_limbs  # Constant number same as max_num_limbs
        self.num_limbs = 1
        self.batch_size = batch_size
        self.obs_padding_mask = self.update_padding_mask()
        self.obs_dim = state_dim
        self.action_dim = action_dim
        self.action_high = max_action

        self.act_net = MetaMorphTransformer(state_dim, action_dim, self.seq_len,
                                            args.attention_embedding_size,
                                            args.attention_heads,
                                            args.attention_hidden_size,
                                            args.attention_layers,
                                            args.dropout_rate)

        self.tanh = nn.Tanh()

    def forward(self, obs, mode="training"):
        trng = mode == "training"
        batch_size = trng * self.batch_size + (1 - trng)
        obs = obs.reshape(batch_size, self.num_limbs, -1)
        pad_size = self.seq_len - self.num_limbs
        obs = F.pad(obs, [0, 0, 0, pad_size])
        obs_mask = self.obs_padding_mask[:batch_size]  # Batch size changes
        act = self.act_net(obs, obs_mask)
        act = self.action_high * self.tanh(act)
        return act[:, :self.num_limbs]

    def change_morphology(self, parents):
        self.num_limbs = len(parents)
        self.obs_padding_mask = self.update_padding_mask()

    def update_padding_mask(self):
        pad_size = self.seq_len - self.num_limbs
        obs_padding_mask = [False] * self.num_limbs + [True] * pad_size
        obs_padding_mask = torch.tensor(obs_padding_mask, device=DEVICE)
        return obs_padding_mask.repeat(self.batch_size, 1)
