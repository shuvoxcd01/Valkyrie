
import tensorflow as tf
from tf_agents.environments import tf_py_environment

from Valkyrie.src.agent.tf_agent.ddqn_agent_factory import DdqnAgentFactory
from Valkyrie.src.checkpoint_manager.agent_checkpoint_manager_factory import \
    AgentCheckpointManagerFactory
from Valkyrie.src.embedding_visualization.embedding_data_manager import \
    EmbeddingDataManager
from Valkyrie.src.environment.pong_factory import PongFactory
from Valkyrie.src.network.atari_q_network_factory import AtariQNetworkFactory

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


env_factory = PongFactory()
eval_py_env = env_factory.get_py_env()
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

observation_spec = eval_env.observation_spec()
action_spec = eval_env.action_spec()
time_step_spec = eval_env.time_step_spec()

network_factory = AtariQNetworkFactory(input_tensor_spec=observation_spec, action_spec=action_spec,
                                       conv_layer_params=CONV_LAYER_PARAMS, fc_layer_params=FC_LAYER_PARAMS)
network = network_factory.get_network()
# optimizer = tf.keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE)
agent_factory = DdqnAgentFactory(time_step_spec=time_step_spec,
                                 action_spec=action_spec, target_update_period=TARGET_UPDATE_PERIOD)

optimizer = tf.compat.v1.train.RMSPropOptimizer(
    learning_rate=2.5e-3,
    decay=0.95,
    momentum=0.0,
    epsilon=0.00001,
    centered=True
)

train_step_counter = tf.Variable(0)
tf_agent = agent_factory.get_agent(name="best", network=network,
                                   optimizer=optimizer,
                                   train_step_counter=train_step_counter)

# base_ckpt_path = "/home/usr/Valkyrie/Valkyrie/training_metadata_pong_v4_2022-05-03 copy/checkpoints"
base_ckpt_path = "/home/Valkyrie/all_training_metadata/pong/training_metadata_pong_v4_2022-05-02 11:41:02.937099/checkpoints"

ckpt_manager = AgentCheckpointManagerFactory(
    base_ckpt_dir=base_ckpt_path).get_agent_checkpoint_manager(tf_agent)

ckpt_manager.create_or_initialize_checkpointer()
# TRAINING_META_DATA_DIR = os.path.join(ALL_TRAINING_METADATA_DIR, "pong",
#                                       "training_metadata_pong_v4_" + str(datetime.now().strftime('%Y-%m-%d-%H.%M.%S')))

TRAINING_META_DATA_DIR = "/home/Valkyrie/all_training_metadata/pong/training_metadata_pong_v4_embedding_test5/"
embedding_data_manager = EmbeddingDataManager(
    base_metadata_dir=TRAINING_META_DATA_DIR)

for _ in range(2):
    time_step = eval_env.reset()
    observation_img = eval_py_env.render()
    features, _network_state = tf_agent._q_network._encoder(
        time_step.observation)
    preds = tf_agent._q_network(time_step.observation)[0].numpy()
    q_value = preds.max()
    embedding_data_manager.add_data(
        img=observation_img, tensor=features, q_value=q_value)
    while not time_step.is_last():
        action_step = tf_agent.policy.action(time_step)
        time_step = eval_env.step(action_step.action)
        observation_img = eval_py_env.render()
        features, _network_state = tf_agent._q_network._encoder(
            time_step.observation)
        preds = tf_agent._q_network(time_step.observation)[0].numpy()
        q_value = preds.max()
        embedding_data_manager.add_data(
            img=observation_img, tensor=features, q_value=q_value)

embedding_data_manager.save()
embedding_data_manager.configure_projector()
