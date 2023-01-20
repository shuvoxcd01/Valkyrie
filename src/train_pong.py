import os
import sys
from datetime import datetime
import tensorflow as tf
from Valkyrie.src.agent.meta_agent.meta_q_agent.meta_q_agent_factory import MetaQAgentFactory
from parent_tracker.parent_tracker import ParentTracker
from fitness_tracker.fitness_tracker import FitnessTracker
from environment.pong_factory import PongFactory
from agent.meta_agent.meta_q_agent.meta_q_agent_copier import MetaQAgentCopier
from checkpoint_manager.agent_checkpoint_manager_factory import AgentCheckpointManagerFactory
from summary_writer.summay_writer_manager_factory import SummaryWriterManagerFactory
from training.population_based_training.population_based_training import PopulationBasedTraining
from driver.py_driver_factory import PyDriverFactory
from fitness_evaluator.fitness_evaluator import FitnessEvaluator
from checkpoint_manager.replay_buffer_checkpoint_manager import ReplayBufferCheckpointManager
from replay_buffer.reverb_replay_buffer_manager import ReverbReplayBufferManager
from training.gradient_based_training.gradient_based_training import GradientBasedTraining
from agent.meta_agent.meta_q_agent.meta_q_agent import MetaQAgent
from network.atari_q_network_factory import AtariQNetworkFactory
from agent.tf_agent.ddqn_agent_factory import DdqnAgentFactory
from tf_agents.environments import tf_py_environment
import logging
from tf_agents.policies import random_py_policy


FC_LAYER_PARAMS = (512,)
CONV_LAYER_PARAMS = ((32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1))
INITIAL_LEARNING_RATE = 2.5e-3  # 1e-3
TARGET_UPDATE_PERIOD = 2000  # 200
REPLAY_BUFFER_MAX_LENGTH = 100000

BATCH_SIZE = 64
LOG_INTERVAL = 1000  # 250
NUM_EVAL_EPISODES = 5
EVAL_INTERVAL = 10000  # 500
INITIAL_COLLECT_STEPS = 200

POPSIZE = 2
NUM_GRADIENT_BASED_TRAINING_EPOCH = 30000
TRAINING_META_DATA_DIR = os.path.join(os.path.dirname(
    os.path.dirname(__file__)), "training_metadata_pong_v4_" + str(datetime.now().strftime('%Y-%m-%d-%H.%M.%S')))

CHECKPOINT_BASE_DIR = os.path.join(TRAINING_META_DATA_DIR, "checkpoints")
SUMMARY_BASE_DIR = os.path.join(TRAINING_META_DATA_DIR, "tf_summaries")
FITNESS_TRACKER_FILE_NAME = "fitness_data.csv"
FITNESS_TRACKER_FILE_PATH = os.path.join(
    TRAINING_META_DATA_DIR, "fitness_tracker", FITNESS_TRACKER_FILE_NAME)

PARENT_TRACKER_FILE_NAME = "parent_data.csv"
PARENT_TRACKER_FILE_PATH = os.path.join(
    TRAINING_META_DATA_DIR, "parent_tracker", PARENT_TRACKER_FILE_NAME)

if not os.path.exists(TRAINING_META_DATA_DIR):
    os.mkdir(TRAINING_META_DATA_DIR)

LOG_FILE_PATH = os.path.join(TRAINING_META_DATA_DIR, "logs.log")

BEST_POSSIBLE_FITNESS = 21
MAX_COLLECT_STEPS = 10
MAX_COLLECT_EPISODES = None


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(LOG_FILE_PATH)
file_handler.setLevel(logging.DEBUG)


logger.addHandler(file_handler)
logger.addHandler(stdout_handler)

initial_population = []

env_factory = PongFactory()

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

meta_agent_factory = MetaQAgentFactory()
meta_agent_copier = MetaQAgentCopier(
    agent_factory=agent_factory, agent_ckpt_manager_factory=agent_checkpoint_manager_factory,
    summary_writer_manager_factory=summary_writer_manager_factory, meta_agent_factory=meta_agent_factory)

for i in range(POPSIZE):
    network = network_factory.get_network()

    optimizer = tf.compat.v1.train.RMSPropOptimizer(
        learning_rate=INITIAL_LEARNING_RATE,
        decay=0.95,
        momentum=0.0,
        epsilon=0.00001,
        centered=True
    )

    train_step_counter = tf.Variable(0)
    tf_agent = agent_factory.get_agent(name="agent_"+str(i), network=network,
                                       optimizer=optimizer,
                                       train_step_counter=train_step_counter)

    agent_checkpoint_manager = agent_checkpoint_manager_factory.get_agent_checkpoint_manager(
        agent=tf_agent)
    summary_writer_manager = summary_writer_manager_factory.get_summary_writer_manager(
        tf_agent=tf_agent)

    meta_agent = MetaQAgent(tf_agent=tf_agent, checkpoint_manager=agent_checkpoint_manager,
                            summary_writer_manager=summary_writer_manager,
                            agent_copier=meta_agent_copier)
    initial_population.append(meta_agent)


fitness_evaluator = FitnessEvaluator(
    environment=eval_env, num_episodes=NUM_EVAL_EPISODES)

collect_data_spec = tf_agent.collect_data_spec

replay_buffer_manager = ReverbReplayBufferManager(
    data_spec=collect_data_spec, replay_buffer_capacity=REPLAY_BUFFER_MAX_LENGTH)

replay_buffer_checkpoint_manager = ReplayBufferCheckpointManager(
    base_ckpt_dir=CHECKPOINT_BASE_DIR,
    replay_buffer=replay_buffer_manager.get_replay_buffer())

replay_buffer_observer = replay_buffer_manager.get_observer()
collect_driver_factory = PyDriverFactory(
    env=train_py_env, observers=[replay_buffer_observer])


random_policy = random_py_policy.RandomPyPolicy(
    time_step_spec=time_step_spec,
    action_spec=action_spec
)

initial_collect_driver = collect_driver_factory.get_driver(
    policy=random_policy, max_steps=INITIAL_COLLECT_STEPS)

gradient_based_trainer = GradientBasedTraining(
    train_env=train_py_env, replay_buffer_manager=replay_buffer_manager,
    replay_buffer_checkpoint_manager=replay_buffer_checkpoint_manager,
    initial_collect_driver=initial_collect_driver,
    fitness_evaluator=fitness_evaluator, num_train_iteration=NUM_GRADIENT_BASED_TRAINING_EPOCH,
    log_interval=LOG_INTERVAL, eval_interval=EVAL_INTERVAL,
    collect_driver_factory=collect_driver_factory, max_collect_steps=MAX_COLLECT_STEPS,
    max_collect_episodes=MAX_COLLECT_EPISODES,
    batch_size=BATCH_SIZE,
    best_possible_fitness=BEST_POSSIBLE_FITNESS)


fitness_tracker = FitnessTracker(csv_file_path=FITNESS_TRACKER_FILE_PATH)
parent_tracker = ParentTracker(csv_file_path=PARENT_TRACKER_FILE_PATH)

population_based_training = PopulationBasedTraining(initial_population=initial_population,
                                                    gradient_based_trainer=gradient_based_trainer,
                                                    fitness_evaluator=fitness_evaluator,
                                                    fitness_trakcer=fitness_tracker,
                                                    parent_tracker=parent_tracker,
                                                    best_possible_fitness=BEST_POSSIBLE_FITNESS,
                                                    num_training_iterations=10
                                                    )

population_based_training.train()
fitness_tracker.plot_fitness()
