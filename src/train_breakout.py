import os
import sys
from datetime import datetime
import tensorflow as tf
from Valkyrie.all_training_metadata import ALL_TRAINING_METADATA_DIR
from Valkyrie.src.agent.meta_agent.meta_q_agent.meta_q_agent_factory import (
    MetaQAgentFactory,
)
from Valkyrie.src.environment.breakout_factory import BreakoutFactory
from Valkyrie.src.network.agent_network.q_networks.atari.atari_q_network_factory import (
    AtariQNetworkFactory,
)
from Valkyrie.src.network.pretraining_network.atari.atari_pretraining_network import (
    AtariPretrainingNetwork,
)
from Valkyrie.src.replay_buffer.unified_replay_buffer.unified_reverb_replay_buffer_manager import (
    UnifiedReverbReplayBufferManager,
)
from Valkyrie.src.training.pretraining.pretraining import Pretraining
from parent_tracker.parent_tracker import ParentTracker
from fitness_tracker.fitness_tracker import FitnessTracker
from agent.meta_agent.meta_q_agent.meta_q_agent_copier import MetaQAgentCopier
from checkpoint_manager.agent_checkpoint_manager_factory import (
    AgentCheckpointManagerFactory,
)
from summary_writer.summay_writer_manager_factory import SummaryWriterManagerFactory
from training.population_based_training.population_based_training import (
    PopulationBasedTraining,
)
from driver.py_driver_factory import PyDriverFactory
from fitness_evaluator.fitness_evaluator import FitnessEvaluator
from checkpoint_manager.replay_buffer_checkpoint_manager import (
    ReplayBufferCheckpointManager,
)
from training.gradient_based_training.gradient_based_training import (
    GradientBasedTraining,
)
from agent.meta_agent.meta_q_agent.meta_q_agent import MetaQAgent
from agent.tf_agent.ddqn_agent_factory import DdqnAgentFactory
from tf_agents.environments import tf_py_environment
import logging
from tf_agents.policies import random_py_policy


FC_LAYER_PARAMS = (256, 128, 64)
CONV_LAYER_PARAMS = ((32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1))
INITIAL_LEARNING_RATE = 2.5e-3  # 1e-3
TARGET_UPDATE_PERIOD = 100  # 200
REPLAY_BUFFER_MAX_LENGTH = 5000

BATCH_SIZE = 64
LOG_INTERVAL = 250  # 250
NUM_EVAL_EPISODES = 1
EVAL_INTERVAL = 250  # 500
INITIAL_COLLECT_STEPS = 1000

POPSIZE = 2
NUM_GRADIENT_BASED_TRAINING_EPOCH = 500
TRAINING_META_DATA_DIR = os.path.join(
    ALL_TRAINING_METADATA_DIR,
    "breakout",
    "training_metadata_breakout_v0_"
    + str(datetime.now().strftime("%Y-%m-%d-%H.%M.%S")),
)

CHECKPOINT_BASE_DIR = os.path.join(TRAINING_META_DATA_DIR, "checkpoints")
SUMMARY_BASE_DIR = os.path.join(TRAINING_META_DATA_DIR, "tf_summaries")
FITNESS_TRACKER_FILE_NAME = "fitness_data.csv"
FITNESS_TRACKER_FILE_PATH = os.path.join(
    TRAINING_META_DATA_DIR, "fitness_tracker", FITNESS_TRACKER_FILE_NAME
)

PARENT_TRACKER_FILE_NAME = "parent_data.csv"
PARENT_TRACKER_FILE_PATH = os.path.join(
    TRAINING_META_DATA_DIR, "parent_tracker", PARENT_TRACKER_FILE_NAME
)

if not os.path.exists(TRAINING_META_DATA_DIR):
    os.mkdir(TRAINING_META_DATA_DIR)

LOG_FILE_PATH = os.path.join(TRAINING_META_DATA_DIR, "logs.log")

BEST_POSSIBLE_FITNESS = None
MAX_COLLECT_STEPS = 10
MAX_COLLECT_EPISODES = None
Q_NETWORK_INITIALIZERS = [tf.keras.initializers.Zeros(), None]

assert (
    len(Q_NETWORK_INITIALIZERS) == POPSIZE
), f"Must provide {POPSIZE} numbers of initializers."


# Pretraining Params
ENCODER_FC_LAYER_PARAMS = (512, 256, 128, 64)
DECODER_FC_LAYER_PARAMS = (128, 256, 512)

NUM_PRETRAINING_ITERATION = 1000
PRETRAINING_BATCH_SIZE = BATCH_SIZE

# Replay Buffer Params
REPLAY_BUFFER_NUM_PARALLEL_CALLS = 2
REPLAY_BUFFER_BATCH_SIZE = BATCH_SIZE
REPLAY_BUFFER_NUM_STEPS = 2
REPLAY_BUFFER_NUM_PREFETCH = 3

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(LOG_FILE_PATH)
file_handler.setLevel(logging.DEBUG)


logger.addHandler(file_handler)
logger.addHandler(stdout_handler)

initial_population = []

env_factory = BreakoutFactory()

train_py_env = env_factory.get_py_env()
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = env_factory.get_tf_env()

train_env_observation_spec = train_env.observation_spec()
action_spec = train_env.action_spec()
time_step_spec = train_env.time_step_spec()

stable_pretraining_network = AtariPretrainingNetwork(
    input_tensor_spec=train_env_observation_spec,
    encoder_fc_layer_params=ENCODER_FC_LAYER_PARAMS,
    decoder_fc_layer_params=DECODER_FC_LAYER_PARAMS,
    encoder_conv_layer_params=None,
    decoder_conv_layer_params=None,
)


running_pretraining_network = AtariPretrainingNetwork(
    input_tensor_spec=train_env_observation_spec,
    encoder_fc_layer_params=ENCODER_FC_LAYER_PARAMS,
    decoder_fc_layer_params=DECODER_FC_LAYER_PARAMS,
    encoder_conv_layer_params=None,
    decoder_conv_layer_params=None,
)

network_factory = AtariQNetworkFactory(
    pretraining_network=stable_pretraining_network,
    input_tensor_spec=stable_pretraining_network.get_encoder_output_spec(),
    action_spec=action_spec,
    conv_layer_params=CONV_LAYER_PARAMS,
    fc_layer_params=FC_LAYER_PARAMS,
)


agent_factory = DdqnAgentFactory(
    time_step_spec=time_step_spec,
    action_spec=action_spec,
    target_update_period=TARGET_UPDATE_PERIOD,
)

agent_checkpoint_manager_factory = AgentCheckpointManagerFactory(
    base_ckpt_dir=CHECKPOINT_BASE_DIR
)
summary_writer_manager_factory = SummaryWriterManagerFactory(
    base_summary_writer_dir=SUMMARY_BASE_DIR
)

meta_agent_factory = MetaQAgentFactory()
meta_agent_copier = MetaQAgentCopier(
    agent_factory=agent_factory,
    agent_ckpt_manager_factory=agent_checkpoint_manager_factory,
    summary_writer_manager_factory=summary_writer_manager_factory,
    meta_agent_factory=meta_agent_factory,
)


for i in range(POPSIZE):
    kernel_initializer = Q_NETWORK_INITIALIZERS[i]
    network = network_factory.get_network(kernel_initializer=kernel_initializer)

    optimizer = tf.keras.optimizers.Adam()

    train_step_counter = tf.Variable(0)
    tf_agent = agent_factory.get_agent(
        name="agent_" + str(i),
        network=network,
        optimizer=optimizer,
        train_step_counter=train_step_counter,
    )

    agent_checkpoint_manager = (
        agent_checkpoint_manager_factory.get_agent_checkpoint_manager(agent=tf_agent)
    )
    summary_writer_manager = summary_writer_manager_factory.get_summary_writer_manager(
        tf_agent=tf_agent
    )

    meta_agent = MetaQAgent(
        tf_agent=tf_agent,
        checkpoint_manager=agent_checkpoint_manager,
        summary_writer_manager=summary_writer_manager,
        agent_copier=meta_agent_copier,
    )
    initial_population.append(meta_agent)


fitness_evaluator = FitnessEvaluator(
    environment=eval_env, num_episodes=NUM_EVAL_EPISODES
)

collect_data_spec = tf_agent.collect_data_spec

replay_buffer_manager = UnifiedReverbReplayBufferManager(
    data_spec=collect_data_spec, replay_buffer_capacity=REPLAY_BUFFER_MAX_LENGTH
)

replay_buffer_checkpoint_manager = ReplayBufferCheckpointManager(
    base_ckpt_dir=CHECKPOINT_BASE_DIR,
    replay_buffer=replay_buffer_manager.get_replay_buffer(),
)

replay_buffer_observer = replay_buffer_manager.get_observer()
collect_driver_factory = PyDriverFactory(
    env=train_py_env, observers=[replay_buffer_observer]
)


random_policy = random_py_policy.RandomPyPolicy(
    time_step_spec=time_step_spec, action_spec=action_spec
)

initial_collect_driver = collect_driver_factory.get_driver(
    policy=random_policy,
    max_steps=INITIAL_COLLECT_STEPS,
)

gradient_based_trainer = GradientBasedTraining(
    train_env=train_py_env,
    replay_buffer_manager=replay_buffer_manager,
    replay_buffer_checkpoint_manager=replay_buffer_checkpoint_manager,
    initial_collect_driver=initial_collect_driver,
    fitness_evaluator=fitness_evaluator,
    num_train_iteration=NUM_GRADIENT_BASED_TRAINING_EPOCH,
    log_interval=LOG_INTERVAL,
    eval_interval=EVAL_INTERVAL,
    collect_driver_factory=collect_driver_factory,
    max_collect_steps=MAX_COLLECT_STEPS,
    max_collect_episodes=MAX_COLLECT_EPISODES,
    batch_size=BATCH_SIZE,
    best_possible_fitness=BEST_POSSIBLE_FITNESS,
)


fitness_tracker = FitnessTracker(csv_file_path=FITNESS_TRACKER_FILE_PATH)
parent_tracker = ParentTracker(csv_file_path=PARENT_TRACKER_FILE_PATH)

pretriner = Pretraining(
    running_pretraining_network=running_pretraining_network,
    stable_pretraining_network=stable_pretraining_network,
    replay_buffer_manager=replay_buffer_manager,
    optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE),
    num_iteration=NUM_PRETRAINING_ITERATION,
    batch_size=PRETRAINING_BATCH_SIZE,
    tf_summary_base_dir=SUMMARY_BASE_DIR,
    tau=0.125,
    stable_network_update_period=500,
)


population_based_training = PopulationBasedTraining(
    initial_population=initial_population,
    pretrainer=pretriner,
    gradient_based_trainer=gradient_based_trainer,
    fitness_evaluator=fitness_evaluator,
    fitness_trakcer=fitness_tracker,
    parent_tracker=parent_tracker,
    best_possible_fitness=BEST_POSSIBLE_FITNESS,
    num_training_iterations=1000,
)

population_based_training.train()
fitness_tracker.plot_fitness()
