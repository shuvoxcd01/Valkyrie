import tensorflow as tf

# tf.config.set_visible_devices([], "GPU")
# tf.debugging.set_log_device_placement(True)
from Valkyrie.src.network.agent_network.q_networks.cartpole.cart_pole_q_network_factory import (
    CartPoleQNetworkFactory,
)
from Valkyrie.src.network.pretraining_network.cartpole.cartpole_pretraining_network import (
    CartPolePretrainingNetwork,
)
from checkpoint_manager.agent_checkpoint_manager_factory import (
    AgentCheckpointManagerFactory,
)
from environment.cartpole_factory import CartPoleFactory
from agent.tf_agent.ddqn_agent_factory import DdqnAgentFactory
from tf_agents.environments import tf_py_environment
import gym


FC_LAYER_PARAMS = (512, 256, 128, 64)
CONV_LAYER_PARAMS = ((32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1))
INITIAL_LEARNING_RATE = 1e-3
TARGET_UPDATE_PERIOD = 50  # 200
REPLAY_BUFFER_MAX_LENGTH = 10000

BATCH_SIZE = 32  # 64
LOG_INTERVAL = 20  # 250
NUM_EVAL_EPISODES = 5
EVAL_INTERVAL = 50  # 500
INITIAL_COLLECT_STEPS = 200

POPSIZE = 1
NUM_GRADIENT_BASED_TRAINING_EPOCH = 100

# Pretraining Params
ENCODER_FC_LAYER_PARAMS = (512, 256, 128, 64)
DECODER_FC_LAYER_PARAMS = (128, 256, 512)

NUM_PRETRAINING_ITERATION = 100
PRETRAINING_BATCH_SIZE = BATCH_SIZE
PRETRAINING_REPLAY_BUFFER_TABLE_NAME = "PRETRAIN"

# Replay Buffer Params
REPLAY_BUFFER_NUM_PARALLEL_CALLS = 2
REPLAY_BUFFER_BATCH_SIZE = BATCH_SIZE
REPLAY_BUFFER_NUM_STEPS = 2
REPLAY_BUFFER_NUM_PREFETCH = 3
REPLAY_BUFFER_TABLE_NAMES = [PRETRAINING_REPLAY_BUFFER_TABLE_NAME]


env_factory = CartPoleFactory()
eval_py_env = env_factory.get_py_env()
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

observation_spec = eval_env.observation_spec()
action_spec = eval_env.action_spec()
time_step_spec = eval_env.time_step_spec()

cartpole_pretraining_network = CartPolePretrainingNetwork(
    input_tensor_spec=observation_spec,
    encoder_fc_layer_params=FC_LAYER_PARAMS,
    decoder_fc_layer_params=DECODER_FC_LAYER_PARAMS,
)

network_factory = CartPoleQNetworkFactory(
    pretraining_network=cartpole_pretraining_network,
    input_tensor_spec=observation_spec,
    action_spec=action_spec,
    conv_layer_params=CONV_LAYER_PARAMS,
    fc_layer_params=FC_LAYER_PARAMS,
)


agent_factory = DdqnAgentFactory(
    time_step_spec=time_step_spec,
    action_spec=action_spec,
    target_update_period=TARGET_UPDATE_PERIOD,
)


network = network_factory.get_network()
optimizer = tf.keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE)
# print(network.get_config())

# optimizer = tf.compat.v1.train.RMSPropOptimizer(
#     learning_rate=INITIAL_LEARNING_RATE,
#     decay=0.95,
#     momentum=0.0,
#     epsilon=0.00001,
#     centered=True,
# )

train_step_counter = tf.Variable(0)
tf_agent = agent_factory.get_agent(
    name="best",
    network=network,
    optimizer=optimizer,
    train_step_counter=train_step_counter,
)

agent_checkpoint_manager_factory = AgentCheckpointManagerFactory(
    base_ckpt_dir="/home/Valkyrie/all_training_metadata/cartpole/training_metadata_cartpole_2023-11-03-12.00.51/checkpoints/best"
)
agent_checkpoint_manager = (
    agent_checkpoint_manager_factory.get_agent_checkpoint_manager(agent=tf_agent)
)

agent_checkpoint_manager.create_or_initialize_checkpointer()

tf_agent._q_network.summary()


for _ in range(10):
    print("============================")
    i = 1
    time_step = eval_env.reset()
    while not time_step.is_last():
        action_step = tf_agent.policy.action(time_step)
        time_step = eval_env.step(action_step.action)
        eval_py_env.render()
        print(i)
        i+=1
        # print(eval_py_env.render())
