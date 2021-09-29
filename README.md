# NeurIPS 2021 IGLU competition RLlib baselines. 

This repository contains an implementation of several baselines for the Builder task of the IGLU contest.
Baselines implemented using RLlib include APE-X, IMPALA, and Rainbow. The code is fully submittable; it contains scripts with a submission-ready agent and for testing a submittable agent.


# Code structure

In `configs/`, you will find configuration files for each baseline. File `train.py` can launch training for each baseline. `wrappers.py` defines all wrappers that are used in training and evaluation. 

File `custom_agent.py` defines a `CusomAgent` class that can load a pretrained model and predict actions given observations from the env. `test_submission.py` holds the code that can test the submission code locally. We recommend you test your submission for bugs as the script contains effectively the same code as in the evaluation server. The current version includes a trained APE-X DQN model with grid+position observations as an example.

# Performance 

This section will be updated with numerical results soon. 

The current baseline works in the single-task setting. 

All RLlib baselines can act in both visual and grid domains. In the current state, given grid input, all models solve simple tasks but fail to solve more complex ones.

# Launch requirements

RLlib baselines mostly use many workers to train. APE-X and IMPALA use 20 workers to train which consumes around 40 CPU cores. With 20 workers, the total memory consumption is 75GB RAM and 6GB of GPU memory. Memory consumption can be reduced by lowering the number of workers, however, models will convegre slower in that case.

# Running the code

We highly recommend running everything from the docker container. Please, use pre-built image `iglucontest/baselines:builder-rllib`.

To run the training e.g. IMPALA, use the following command:

```
python train.py -f configs/impala.yml
```

# How to submit a model

Before submitting, make sure to test your code locally.
To test the submission, provide a valid path to a saved checkpoint in `custom_agent.py` and run

```
python test_submission.py
```

If you want your image to be used in the evaluation, put the name of your image into `Dockerimage`. Also, you should install the following mandatory **PYTHON PACKAGES** that are used in the evaluation server:
`docker==5.0.2 requests==2.26.0 tqdm==4.62.3 pyyaml==5.4.1 gym==0.18.3`.

To submit the model, just zip the whole directory (do not forget to add recursive paths) and submit the zip file in the [codalab competition](https://competitions.codalab.org/competitions/33828).

# Baseline summary

Environment wrappers:

  * We unify block selection actions and placement actions. That is, each Hotbar selection will be immediately followed by the placement of the chosen block.
  * After each place/break action, we do **exactly** 3 noop actions to wait until the, e.g., break event will happen. This is due to how Minecraft processes actions internally.
  * The actions are discretized into a simple categorical space. There are four movement actions (forth, back, left, right); 4 camera actions (look up, look down, look left or right); each camera action changes one angle by 5 degrees. Jump action, break action, 6 Hotbar selection/placement actions, and one noop action. In total, there are 17 discrete actions.
  * We change the reward function; the new one tracks the maximal intersection size of two zones and gives only `+1` iff it increased a maximum reached size during the episode. This means that the maximal possible episode return is equal to the size of the target structure.