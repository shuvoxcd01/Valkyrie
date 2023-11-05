import tensorflow as tf
from tf_agents.environments import tf_py_environment

from Valkyrie.src.agent.tf_agent.ddqn_agent_factory import DdqnAgentFactory
from Valkyrie.src.checkpoint_manager.agent_checkpoint_manager_factory import (
    AgentCheckpointManagerFactory,
)
from Valkyrie.src.embedding_visualization.embedding_data_manager import (
    EmbeddingDataManager,
)
from Valkyrie.src.environment.pong_factory import PongFactory
from Valkyrie.src.fitness_evaluator.fitness_evaluator import FitnessEvaluator
from Valkyrie.src.network.agent_network.q_networks.atari.atari_q_network_factory import (
    AtariQNetworkFactory,
)
from Valkyrie.src.network.pretraining_network.atari.atari_pretraining_network import (
    AtariPretrainingNetwork,
)

FC_LAYER_PARAMS = (256, 128, 64)
CONV_LAYER_PARAMS = ((32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1))
INITIAL_LEARNING_RATE = 2.5e-3  # 1e-3
TARGET_UPDATE_PERIOD = 100  # 200
REPLAY_BUFFER_MAX_LENGTH = 10000

BATCH_SIZE = 64
LOG_INTERVAL = 1000  # 250
NUM_EVAL_EPISODES = 5
EVAL_INTERVAL = 10000  # 500
INITIAL_COLLECT_STEPS = 200

POPSIZE = 2
NUM_GRADIENT_BASED_TRAINING_EPOCH = 30000
ENCODER_FC_LAYER_PARAMS = (512, 256, 128, 64)
DECODER_FC_LAYER_PARAMS = (128, 256, 512)

env_factory = PongFactory()
eval_py_env = env_factory.get_py_env()
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

observation_spec = eval_env.observation_spec()
action_spec = eval_env.action_spec()
time_step_spec = eval_env.time_step_spec()
stable_pretraining_network = AtariPretrainingNetwork(
    input_tensor_spec=observation_spec,
    encoder_fc_layer_params=ENCODER_FC_LAYER_PARAMS,
    decoder_fc_layer_params=DECODER_FC_LAYER_PARAMS,
    encoder_conv_layer_params=None,
    decoder_conv_layer_params=None,
)
network_factory = AtariQNetworkFactory(
    pretraining_network=stable_pretraining_network,
    input_tensor_spec=observation_spec,
    action_spec=action_spec,
    conv_layer_params=CONV_LAYER_PARAMS,
    fc_layer_params=FC_LAYER_PARAMS,
)
network = network_factory.get_network()
agent_factory = DdqnAgentFactory(
    time_step_spec=time_step_spec,
    action_spec=action_spec,
    target_update_period=TARGET_UPDATE_PERIOD,
)

optimizer = tf.keras.optimizers.Adam()

train_step_counter = tf.Variable(0)
tf_agent = agent_factory.get_agent(
    name="best",
    network=network,
    optimizer=optimizer,
    train_step_counter=train_step_counter,
)

base_ckpt_path = "/home/Valkyrie/all_training_metadata/pong/training_metadata_pong_v4_2023-11-04-13.56.03/checkpoints"

ckpt_manager = AgentCheckpointManagerFactory(
    base_ckpt_dir=base_ckpt_path
).get_agent_checkpoint_manager(tf_agent)

ckpt_manager.create_or_initialize_checkpointer()

tf_agent._q_network.summary()


for _ in range(20):
    print("============================")
    i = 1
    time_step = eval_env.reset()
    while not time_step.is_last():
        action_step = tf_agent.policy.action(time_step)
        time_step = eval_env.step(action_step.action)
        observation_img = eval_py_env.render("human")
