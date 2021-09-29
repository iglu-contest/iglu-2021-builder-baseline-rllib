import yaml
import ray
import os
import gym
import iglu
import sys
import wandb
import logging
from collections import defaultdict
from filelock import FileLock
from iglu.tasks import RandomTasks, TaskSet
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.integration.wandb import WandbLogger
from argparse import ArgumentParser
from wrappers import \
    SelectAndPlace, \
    Discretization, \
    flat_action_space, \
    SizeReward, \
    TimeLimit, \
    VectorObservationWrapper, \
    VisualObservationWrapper, \
    Logger
from ray.rllib.evaluation.metrics import collect_episodes, summarize_episodes

from model import GridBaselineModel, PovBaselineModel

logging.basicConfig(stream=sys.stdout)


def evaluate_separately(trainer, eval_workers):
    w = next(iter(eval_workers.remote_workers()))
    env_ids = ray.get(w.foreach_env.remote(lambda env: list(env.tasks.preset.keys())))[0]
    print(f'env id: {env_ids}')
    i = 0
    all_episodes = []
    while i < len(env_ids):
        for w in eval_workers.remote_workers():
            w.foreach_env.remote(lambda env: env.set_task(env_ids[i]))
            i += 1
        ray.get([w.sample.remote() for w in eval_workers.remote_workers()])
        episodes, _ = collect_episodes(
            remote_workers=eval_workers.remote_workers(), timeout_seconds=99999)
        all_episodes += episodes
    metrics = summarize_episodes(episodes)
    for eid, ep in zip(env_ids, all_episodes):
        metrics[f'env_{eid}_reward'] = ep.episode_reward
    return metrics


def build_env(env_config=None, env_factory=None):
    """

    Args:
        env_config (dict): a dictionary with following keys:
            * action_space :: human-level | discrete | continuous
            * visual :: (bool) whether to expose only visual observation
            * size_reward :: (bool) whether to use reward for increasing size, otherwise default
            * task_mode :: possible values are: 'one_task', 'many_tasks', 'random_tasks' 
                if task_mode is one_task -> string with task id
                if task_mode is many_tasks -> list of task ids
                if task_mode is random_tasks -> ignored
            * task_id :: (str or list[str]) task id list of task ids
            * random_tasks :: specification for the random tasks generator. for details,
                see the documentation of iglu.tasks.RandomTasks
        env_factory (callable, optional): function that returns a env instance
    
    """
    import iglu
    from iglu.tasks import TaskSet
    if env_config is None:
        env_config = defaultdict(lambda: defaultdict(dict))
    if env_factory is None:
        env = gym.make('IGLUSilentBuilder-v0', max_steps=5000)
        if env_config['task_mode'] == 'one_task':
            env.update_taskset(TaskSet(preset=[env_config['task_id']]))
            env.set_task(env_config['task_id'])
        elif env_config['task_mode'] == 'many_tasks':
            env.update_taskset(TaskSet(preset=env_config['task_id']))
        elif env_config['task_mode'] == 'random_tasks':
            env.update_taskset(RandomTasks(
               max_blocks=env_config['random_tasks'].get('max_blocks', 3),
               height_levels=env_config['random_tasks'].get('height_levels', 1),
               allow_float=env_config['random_tasks'].get('allow_float', False),
               max_dist=env_config['random_tasks'].get('max_dist', 2),
               num_colors=env_config['random_tasks'].get('num_colors', 1),
               max_cache=env_config['random_tasks'].get('max_cache', 0),
            ))
    else:
        env = env_factory()
    #env = Logger(env)
    env = SelectAndPlace(env)
    env = Discretization(env, flat_action_space(env_config['action_space']))
    # visual - pov + inventory + compass + target grid; 
    # vector: grid + position + inventory + target grid
    if env_config['visual']:
        env = VisualObservationWrapper(env)
    else:
        env = VectorObservationWrapper(env)
    if env_config.get('size_reward', False):
        env = SizeReward(env)
    env = TimeLimit(env, limit=env_config['time_limit'])
    return env

def register_models():
    ModelCatalog.register_custom_model(
        "grid_baseline_model", GridBaselineModel)
    ModelCatalog.register_custom_model(
        "pov_baseline_model", PovBaselineModel)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-f', type=str, help='file')
    parser.add_argument('--local', action='store_true', default=False)
    parser.add_argument('--wdb', action='store_true', default=False)
    args = parser.parse_args()
    if args.local:
        ray.init(local_mode=True)
    tune.register_env('IGLUSilentBuilder-v0', build_env)
    register_models()

    with open(args.f) as f:
        config = yaml.load(f)
    for key in config:
        if args.wdb:
            config[key]['config']['logger_config'] = {}
            config[key]['config']['logger_config']['wandb'] = {
                "api_key": os.environ.get('WANDB_APIKEY'),
                "project": key,
                "log_config": False
            }
        config[key]['config']['env'] = config[key]['env']
        run = config[key]['run']
        print(config)
        del config[key]['env'], config[key]['run']
        config[key]['config']['custom_eval_function'] = evaluate_separately
        if args.local:
            config[key]['config']['num_workers'] = 1
            config[key]['stop']['timesteps_total'] = 3000
            config[key]['config']['timesteps_per_iteration'] = 100
            # config[key]['config']['learning_starts'] = 0
            # if args.wdb:
            #     del config[key]['config']['logger_config']['wandb']
        if args.wdb:
            loggers = DEFAULT_LOGGERS + (WandbLogger, )
        else:
            loggers = DEFAULT_LOGGERS
            
        tune.run(run, **config[key], loggers=loggers)
