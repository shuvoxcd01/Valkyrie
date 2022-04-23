from pickletools import optimize
from agent.tf_agent.ddqn_agent_factory import DdqnAgentFactory
from checkpoint_manager.replay_buffer_checkpoint_manager import ReplayBufferCheckpointManager
from environment.cartpole_factory import CartPoleFactory
from driver.py_driver_factory import PyDriverFactory
from replay_buffer.reverb_replay_buffer_manager import ReverbReplayBufferManager
from agent.tf_agent.dqn_agent_factory import DqnAgentFactory
from driver.dynamic_step_driver_factory import DynamicStepDriverFactory
from environment.pong_factory import PongFactory
from network.atari_q_network_factory import AtariQNetworkFactory
from replay_buffer.replay_buffer_manager import ReplayBufferManager
from training.gradient_based_training.gradient_based_training import GradientBasedTraining
import tensorflow as tf
from tf_agents.policies import random_tf_policy, random_py_policy, py_tf_eager_policy
from tf_agents.utils import common
import os
from tf_agents.environments import suite_atari, suite_gym, tf_py_environment, batched_py_environment, parallel_py_environment


NUM_ITERATIONS = 250000

INITIAL_COLLECT_STEPS = 100
COLLECT_STEPS_PER_ITERATION = 1
REPLAY_BUFFER_MAX_LENGTH = 100000

BATCH_SIZE = 64
LEARNING_RATE = 0.001
LOG_INTERVAL = 200

NUM_EVAL_EPISODES = 10
EVAL_INTERVAL = 1000
TARGET_UPDATE_PERIOD = 50

FC_LAYER_PARAMS = (512,)
CONV_LAYER_PARAMS = None  # ((32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1))

env_factory = CartPoleFactory()

train_py_env = env_factory.get_py_env()
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = env_factory.get_tf_env()

train_env_observation_spec = train_env.observation_spec()
# assert train_env.observation_spec() == eval_env.observation_spec()
action_spec = train_env.action_spec()
time_step_spec = train_env.time_step_spec()


network_factory = AtariQNetworkFactory(input_tensor_spec=train_env_observation_spec,
                                       action_spec=action_spec, conv_layer_params=CONV_LAYER_PARAMS,
                                       fc_layer_params=FC_LAYER_PARAMS)
network = network_factory.get_network()

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

agent_factory = DdqnAgentFactory(
    time_step_spec=time_step_spec, action_spec=action_spec,
    q_network=network, optimizer=optimizer, target_update_period=TARGET_UPDATE_PERIOD,
    train_step_counter=tf.Variable(0))

agent = agent_factory.get_agent()
# agent_policy = py_tf_eager_policy.PyTFEagerPolicy(
#     agent.policy, use_tf_function=True)
# agent_collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
#     agent.collect_policy, use_tf_function=True)

agent_py_policy = py_tf_eager_policy.PyTFEagerPolicy(
    agent.policy, use_tf_function=True)
agent_py_collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
    agent.collect_policy, use_tf_function=True)

print(agent.name)

replay_buffer_manager = ReverbReplayBufferManager(
    data_spec=agent.collect_data_spec, replay_buffer_capacity=REPLAY_BUFFER_MAX_LENGTH)
replay_buffer_observer = replay_buffer_manager.get_observer()

replay_buffer_checkpoint_manager = ReplayBufferCheckpointManager(base_ckpt_dir=os.path.join(os.path.dirname(__file__), "checkpoint"))
replay_buffer_checkpoint_manager.create_or_initialize_checkpointer(replay_buffer_manager.get_replay_buffer())


driver_factory = PyDriverFactory(
    env=train_py_env, observers=[replay_buffer_observer])

random_policy = random_py_policy.RandomPyPolicy(
    time_step_spec=time_step_spec,
    action_spec=action_spec
)

initial_collect_driver = driver_factory.get_driver(
    policy=random_policy, max_steps=INITIAL_COLLECT_STEPS)

collect_driver = driver_factory.get_driver(
    policy=agent_py_collect_policy, max_steps=COLLECT_STEPS_PER_ITERATION)

agent_checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoint")

agent_checkpointer = common.Checkpointer(
    ckpt_dir=agent_checkpoint_dir,
    max_to_keep=1,
    agent=agent,
    policy=agent.policy,
    replay_buffer=replay_buffer_manager.replay_buffer
)

training = GradientBasedTraining(train_env=train_py_env, eval_env=eval_env,
                    replay_buffer_manager=replay_buffer_manager, base_summary_writer_dir="./logs", base_checkpoint_dir=None)

initial_collect_driver.run(train_py_env.reset())

training.train_agent(num_iterations=NUM_ITERATIONS, num_eval_episodes=NUM_EVAL_EPISODES,
                     log_interval=LOG_INTERVAL, eval_interval=EVAL_INTERVAL, batch_size=BATCH_SIZE)
