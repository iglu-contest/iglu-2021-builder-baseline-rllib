from typing import Sequence
import gym
import numpy as np
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.modules.noisy_layer import NoisyLayer
from ray.rllib.agents.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict
from ray.rllib.utils.annotations import override

from torch import nn
from ray.rllib.models.preprocessors import DictFlatteningPreprocessor, get_preprocessor
torch, nn = try_import_torch()


class PovBaselineModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs,
                              model_config, name)
        if num_outputs is None:
            # required by rllib's lstm wrapper
            num_outputs = int(np.product(self.obs_space.shape))
        pov_embed_size = 256
        inv_emded_size = 256
        embed_size = 512
        self.pov_embed = nn.Sequential(
            nn.Conv2d(3, 64, 4, 4),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 4),
            nn.ReLU(),
            nn.Conv2d(128, pov_embed_size, 4, 4),
            nn.ReLU(),
        )
        self.inventory_compass_emb = nn.Sequential(
            nn.Linear(7, inv_emded_size),
            nn.ReLU(),
            nn.Linear(inv_emded_size, inv_emded_size),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(pov_embed_size + inv_emded_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, embed_size),
            nn.ReLU(),
            nn.Linear(embed_size, num_outputs),
        )

    def forward(self, input_dict, state, seq_lens):
        obs = input_dict['obs']
        pov = obs['pov'] / 255. - 0.5
        pov = pov.transpose(2, 3).transpose(1, 2).contiguous()
        pov_embed = self.pov_embed(pov)
        pov_embed = pov_embed.reshape(pov_embed.shape[0], -1)

        inventory_compass = torch.cat([obs['inventory'], obs['compass']], 1)
        inv_comp_emb = self.inventory_compass_emb(inventory_compass)

        head_input = torch.cat([pov_embed, inv_comp_emb], 1)
        return self.head(head_input), state


class GridBaselineModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name):
        # flat_obs = {o: obs_space[o] for o in ['agentPos', 'inventory']}
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs,
                              model_config, name)
        if num_outputs is None:
            # required by rllib's lstm wrapper
            num_outputs = int(np.product(self.obs_space.shape))
        hidden_grid = 300
        hidden_vec = 300
        hidden = 300
        self.encode_grid = nn.Sequential(
            nn.Linear(9*11*11, hidden_grid),
            nn.ReLU(),
            nn.Linear(hidden_grid, hidden_grid),
            nn.ReLU(),
            nn.Linear(hidden_grid, hidden_grid),
            nn.ReLU(),
            nn.Linear(hidden_grid, hidden_grid),
        )
        self.encode_pos_inventory = nn.Sequential(
            nn.Linear(11, hidden_vec),
            nn.ReLU(),
            nn.Linear(hidden_vec, hidden_vec),
            nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.Linear(2*hidden_grid + hidden_vec, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_outputs)
        )
        
    def forward(self, input_dict, state, seq_lens):
        grid = input_dict['obs']['grid']
        target_grid = input_dict['obs']['target_grid']
        grid = grid.reshape(grid.shape[0], -1)
        target_grid = target_grid.reshape(target_grid.shape[0], -1)
        vector_input = torch.cat([input_dict['obs']['agentPos'], input_dict['obs']['inventory']], -1)

        grid_embed = self.encode_grid(grid)
        target_grid_embed = self.encode_grid(target_grid)
        vec_embed = self.encode_pos_inventory(vector_input)

        head_input = torch.cat([grid_embed, target_grid_embed, vec_embed], -1)

        return self.head(head_input), state