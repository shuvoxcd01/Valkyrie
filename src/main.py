from pickletools import optimize
from agent.dqn_agent_factory import DqnAgentFactory
from driver.dynamic_step_driver_factory import DynamicStepDriverFactory
from environment.pong_factory import PongFactory
from network.atari_q_network_factory import AtariQNetworkFactory
from replay_buffer.replay_buffer_manager import ReplayBufferManager
from training.training import Training
import tensorflow as tf
from tf_agents.policies import random_tf_policy

NUM_ITERATIONS = 250000

INITIAL_COLLECT_STEPS = 200
COLLECT_STEPS_PER_ITERATION = 10
REPLAY_BUFFER_MAX_LENGTH = 100000

BATCH_SIZE = 32
LEARNING_RATE = 2.5e-3
LOG_INTERVAL = 5000

NUM_EVAL_EPISODES = 10
EVAL_INTERVAL = 25000
TARGET_UPDATE_PERIOD = 2000

FC_LAYER_PARAMS = (512,)
CONV_LAYER_PARAMS = ((32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1))

env_factory = PongFactory()
train_env = env_factory.get_tf_env()
eval_env = env_factory.get_tf_env()
train_env_observation_spec = train_env.observation_spec()
action_spec = train_env.action_spec()
time_step_spec = train_env.time_step_spec()

replay_buffer_manager = ReplayBufferManager()
replay_buffer_observer = replay_buffer_manager.get_observer()
network_factory = AtariQNetworkFactory(input_tensor_spec=train_env_observation_spec,
                                       action_spec=action_spec, conv_layer_params=CONV_LAYER_PARAMS,
                                       fc_layer_params=FC_LAYER_PARAMS)
network = network_factory.get_network()

optimizer = tf.keras.optimizers.RMSPropOptimizer(
    learning_rate=LEARNING_RATE,
    decay=0.95,
    momentum=0.0,
    epsilon=0.00001,
    centered=True
)

agent_factory = DqnAgentFactory(
    time_step_spec=time_step_spec, action_spec=action_spec,
    q_network=network, optimizer=optimizer, target_update_period=TARGET_UPDATE_PERIOD,
    train_step_counter=tf.Variable(0))

agent = agent_factory.get_agent()

driver_factory = DynamicStepDriverFactory(
    env=train_env, observers=[replay_buffer_observer])

random_policy = random_tf_policy.RandomTFPolicy(
    time_step_spec=time_step_spec,
    action_spec=action_spec
)

initial_collect_driver = driver_factory.get_driver(
    policy=random_policy, num_steps=INITIAL_COLLECT_STEPS)
collect_driver = driver_factory.get_driver(
    policy=agent.collect_policy, num_steps=COLLECT_STEPS_PER_ITERATION)

training = Training(agent=agent, collect_driver=collect_driver, eval_env=eval_env, replay_buffer_manager=replay_buffer_manager, logdir="./logs")
