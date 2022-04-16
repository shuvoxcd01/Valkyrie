import os
import tensorflow as tf
from checkpoint_manager.agent_checkpoint_manager_factory import AgentCheckpointManagerFactory
from summary_writer.summary_writer_manager import SummaryWriterManager
from summary_writer.summay_writer_manager_factory import SummaryWriterManagerFactory
from training.population_based_training.population_based_training import PopulationBasedTraining
from driver.py_driver_factory import PyDriverFactory
from fitness_evaluator.fitness_evaluator import FitnessEvaluator
from checkpoint_manager.replay_buffer_checkpoint_manager import ReplayBufferCheckpointManager
from replay_buffer.reverb_replay_buffer_manager import ReverbReplayBufferManager
from training.gradient_based_training.gradient_based_training import GradientBasedTraining
from agent.meta_agent.meta_agent import MetaAgent
from checkpoint_manager.agent_checkpoint_manager import AgentCheckpointManager
from network.atari_q_network_factory import AtariQNetworkFactory
from environment.cartpole_factory import CartPoleFactory
from agent.tf_agent.ddqn_agent_factory import DdqnAgentFactory
from tf_agents.environments import suite_atari, suite_gym, tf_py_environment, batched_py_environment, parallel_py_environment
import logging
from tf_agents.policies import random_tf_policy, random_py_policy, py_tf_eager_policy


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

FC_LAYER_PARAMS = (512,)
CONV_LAYER_PARAMS = None  # ((32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1))
INITIAL_LEARNING_RATE = 0.001
TARGET_UPDATE_PERIOD = 50
REPLAY_BUFFER_MAX_LENGTH = 100000

BATCH_SIZE = 64
LOG_INTERVAL = 200
NUM_EVAL_EPISODES = 10
EVAL_INTERVAL = 1000
INITIAL_COLLECT_STEPS = 100

POPSIZE = 5
NUM_GRADIENT_BASED_TRAINING_EPOCH = 20000
CHECKPOINT_BASE_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
SUMMARY_BASE_DIR = os.path.join(os.path.dirname(__file__), "logs")

initial_population = []

env_factory = CartPoleFactory()

train_py_env = env_factory.get_py_env()
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = env_factory.get_tf_env()

train_env_observation_spec = train_env.observation_spec()
action_spec = train_env.action_spec()
time_step_spec = train_env.time_step_spec()


network_factory = AtariQNetworkFactory(input_tensor_spec=train_env_observation_spec,
                                       action_spec=action_spec, conv_layer_params=CONV_LAYER_PARAMS,
                                       fc_layer_params=FC_LAYER_PARAMS)

agent_factory = DdqnAgentFactory(time_step_spec=time_step_spec,
                                 action_spec=action_spec, target_update_period=TARGET_UPDATE_PERIOD)

agent_checkpoint_manager_factory = AgentCheckpointManagerFactory(
    base_ckpt_dir=CHECKPOINT_BASE_DIR)
summary_writer_manager_factory = SummaryWriterManagerFactory(
    base_summary_writer_dir=SUMMARY_BASE_DIR)

for i in range(POPSIZE):
    network = network_factory.get_network()
    optimizer = tf.keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE)
    train_step_counter = tf.Variable(0)
    tf_agent = agent_factory.get_agent(name="agent_"+str(i), network=network,
                                       optimizer=optimizer,
                                       train_step_counter=train_step_counter)

    agent_checkpoint_manager = agent_checkpoint_manager_factory.get_agent_checkpoint_manager(
        agent=tf_agent)
    summary_writer_manager = summary_writer_manager_factory.get_summary_writer_manager(
        tf_agent=tf_agent)

    meta_agent = MetaAgent(tf_agent=tf_agent, checkpoint_manager=agent_checkpoint_manager,
                           summary_writer_manager=summary_writer_manager)
    initial_population.append(meta_agent)


collect_data_spec = tf_agent.collect_data_spec

replay_buffer_manager = ReverbReplayBufferManager(
    data_spec=collect_data_spec, replay_buffer_capacity=REPLAY_BUFFER_MAX_LENGTH)


replay_buffer_checkpoint_manager = ReplayBufferCheckpointManager(
    base_ckpt_dir=CHECKPOINT_BASE_DIR,
    replay_buffer=replay_buffer_manager.get_replay_buffer())

fitness_evaluator = FitnessEvaluator(
    environment=eval_env, num_episodes=NUM_EVAL_EPISODES)

replay_buffer_observer = replay_buffer_manager.get_observer()
collect_driver_factory = PyDriverFactory(
    env=train_py_env, observers=[replay_buffer_observer])

random_policy = random_py_policy.RandomPyPolicy(
    time_step_spec=time_step_spec,
    action_spec=action_spec
)

initial_collect_driver = collect_driver_factory.get_driver(
    policy=random_policy, num_steps=INITIAL_COLLECT_STEPS)

gradient_based_trainer = GradientBasedTraining(
    train_env=train_py_env, replay_buffer_manager=replay_buffer_manager,
    replay_buffer_checkpoint_manager=replay_buffer_checkpoint_manager,
    initial_collect_driver=initial_collect_driver,
    fitness_evaluator=fitness_evaluator, num_train_iteration=NUM_GRADIENT_BASED_TRAINING_EPOCH,
    log_interval=LOG_INTERVAL, eval_interval=EVAL_INTERVAL, batch_size=BATCH_SIZE)


population_based_training = PopulationBasedTraining(initial_population=initial_population,
                                                    gradient_based_trainer=gradient_based_trainer,
                                                    agent_factory=agent_factory,
                                                    fitness_evaluator=fitness_evaluator,
                                                    collect_driver_factory=collect_driver_factory,
                                                    agent_ckpt_manager_factory=agent_checkpoint_manager_factory,
                                                    summary_writer_manager_factory=summary_writer_manager_factory)

population_based_training.train()
