import tensorflow as tf
import torch 
import gym
from copy import deepcopy as copy
from gym import spaces
import ray
import numpy as np
from torch._C import Value
import yaml


from wrappers import FakeIglu
from train import build_env, register_models
from ray.rllib.agents.registry import get_trainer_class


CONFIG_FILE = './apex_c32/apex_c32.yml'


class CustomAgent:
    def __init__(self, action_space):
        ray.init(local_mode=True)
        self.action_space = action_space
        with open(CONFIG_FILE, 'r') as f:
            config = yaml.safe_load(f)['iglu-baseline']
        with open('metadata', 'r') as f:
            meta = yaml.safe_load(f)
        if meta['action_space'] != config['config']['env_config']['action_space']:
            metadata_as = meta['action_space']
            model_as = config['config']['env_config']['action_space']
            raise ValueError(
                'requested action space in metadata file differs '
                'from the one selected by the model. '
                f'Metadata action space: {metadata_as}; Model action space {model_as}')
        register_models()
        Trainer = get_trainer_class(config['run'])
        self.config = config
        config['config']['in_evaluation'] = True
        self.fake_env = build_env(
            env_config=config['config']['env_config'],
            env_factory=lambda: FakeIglu(config['config']['env_config'], wrap_actions=False)
        )
        self.visual = config['config']['env_config']['visual']
        agent = Trainer(config=config['config'], env=FakeIglu)
        agent.restore('./apex_c32/apex_c32')
        self.agent = agent
        self.actions = iter([])
        self.state = None

    def policy(self, obs, reward, done, info, state):
        if self.agent.config['model'].get('use_lstm', False) and state is None:
            cell_size = self.agent.config['model'].get('lstm_cell_size')
            state = [
                torch.zeros((cell_size,)).float(),
                torch.zeros((cell_size,)).float(),
            ]
        output = self.agent.compute_single_action(
            obs, explore=False, state=state
        )
        if not isinstance(output, tuple):
            action = output
        else:
            action, state, _ = output
        return action, state
        
    def act(self, obs, reward, done, info):
        if done:
            self.actions = iter([])
            self.state = None
            return
        try:
            action = next(self.actions)
        except StopIteration:
            obs = self.fake_env.wrap_observation(obs, reward, done, info)
            agent_action, self.state = self.policy(obs, reward, done, info, self.state)
            self.actions = iter(self.fake_env.stack_actions()(agent_action))
            action = next(self.actions)
        return copy(action)
