from pickletools import optimize
from driver.py_driver_factory import PyDriverFactory
from replay_buffer.reverb_replay_buffer_manager import ReverbReplayBufferManager
from agent.dqn_agent_factory import DqnAgentFactory
from driver.dynamic_step_driver_factory import DynamicStepDriverFactory
from environment.pong_factory import PongFactory
from network.atari_q_network_factory import AtariQNetworkFactory
from replay_buffer.replay_buffer_manager import ReplayBufferManager
from training.training import Training
import tensorflow as tf
from tf_agents.policies import random_tf_policy, random_py_policy, py_tf_eager_policy

NUM_ITERATIONS = 250000

INITIAL_COLLECT_STEPS = 200
COLLECT_STEPS_PER_ITERATION = 10
REPLAY_BUFFER_MAX_LENGTH = 1000

BATCH_SIZE = 32
LEARNING_RATE = 2.5e-3
LOG_INTERVAL = 5000

NUM_EVAL_EPISODES = 10
EVAL_INTERVAL = 25000
TARGET_UPDATE_PERIOD = 2000

FC_LAYER_PARAMS = (512,)
CONV_LAYER_PARAMS = ((32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1))

env_factory = PongFactory()
train_env = env_factory.get_py_env()
eval_env = env_factory.get_py_env()
train_env_observation_spec = train_env.observation_spec()
action_spec = train_env.action_spec()
time_step_spec = train_env.time_step_spec()


network_factory = AtariQNetworkFactory(input_tensor_spec=train_env_observation_spec,
                                       action_spec=action_spec, conv_layer_params=CONV_LAYER_PARAMS,
                                       fc_layer_params=FC_LAYER_PARAMS)
network = network_factory.get_network()

optimizer = tf.keras.optimizers.RMSprop(
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
agent_policy = py_tf_eager_policy.PyTFEagerPolicy(
    agent.policy, use_tf_function=True)
agent_collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
    agent.collect_policy, use_tf_function=True)

replay_buffer_manager = ReverbReplayBufferManager(
    data_spec=agent.collect_data_spec, replay_buffer_capacity=REPLAY_BUFFER_MAX_LENGTH)
replay_buffer_observer = replay_buffer_manager.get_observer()

driver_factory = PyDriverFactory(
    env=train_env, observers=[replay_buffer_observer])

random_policy = random_py_policy.RandomPyPolicy(
    time_step_spec=time_step_spec,
    action_spec=action_spec
)

initial_collect_driver = driver_factory.get_driver(
    policy=random_policy, num_steps=INITIAL_COLLECT_STEPS)

collect_driver = driver_factory.get_driver(
    policy=agent_collect_policy, num_steps=COLLECT_STEPS_PER_ITERATION)

training = Training(agent=agent, collect_driver=collect_driver, train_env=train_env, eval_env=eval_env,
                    replay_buffer_manager=replay_buffer_manager, logdir="./logs")

initial_collect_driver.run(train_env.reset())

training.train_agent(num_iterations=10, num_eval_episodes=2,
                     log_interval=1, eval_interval=1, batch_size=4)
