import tensorflow as tf
from network.cart_pole_q_network_factory import CartPoleQNetworkFactory
from checkpoint_manager.agent_checkpoint_manager_factory import AgentCheckpointManagerFactory
from environment.cartpole_factory import CartPoleFactory
from agent.tf_agent.ddqn_agent_factory import DdqnAgentFactory
from tf_agents.environments import tf_py_environment
import gym


FC_LAYER_PARAMS = (512,)
CONV_LAYER_PARAMS = ((32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1))
INITIAL_LEARNING_RATE = 2.5e-3  # 1e-3
TARGET_UPDATE_PERIOD = 50  # 200
REPLAY_BUFFER_MAX_LENGTH = 100000

BATCH_SIZE = 32  # 64
LOG_INTERVAL = 20  # 250
NUM_EVAL_EPISODES = 5
EVAL_INTERVAL = 50  # 500
INITIAL_COLLECT_STEPS = 200

POPSIZE = 2
NUM_GRADIENT_BASED_TRAINING_EPOCH = 100


env_factory = CartPoleFactory()
eval_py_env = env_factory.get_py_env()
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

observation_spec = eval_env.observation_spec()
action_spec = eval_env.action_spec()
time_step_spec = eval_env.time_step_spec()


network_factory = CartPoleQNetworkFactory(input_tensor_spec=observation_spec,
                                          action_spec=action_spec, conv_layer_params=CONV_LAYER_PARAMS,
                                          fc_layer_params=FC_LAYER_PARAMS)


agent_factory = DdqnAgentFactory(time_step_spec=time_step_spec,
                                 action_spec=action_spec, target_update_period=TARGET_UPDATE_PERIOD)


network = network_factory.get_network()
# optimizer = tf.keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE)
# print(network.get_config())

optimizer = tf.compat.v1.train.RMSPropOptimizer(
    learning_rate=INITIAL_LEARNING_RATE,
    decay=0.95,
    momentum=0.0,
    epsilon=0.00001,
    centered=True
)

train_step_counter = tf.Variable(0)
tf_agent = agent_factory.get_agent(name="best", network=network,
                                   optimizer=optimizer,
                                   train_step_counter=train_step_counter)

agent_checkpoint_manager_factory = AgentCheckpointManagerFactory(
    base_ckpt_dir="/home/Valkyrie/training_metadata_cartpole_2022-12-10-04.05.20/checkpoints")
agent_checkpoint_manager = agent_checkpoint_manager_factory.get_agent_checkpoint_manager(
    agent=tf_agent)

agent_checkpoint_manager.create_or_initialize_checkpointer()

tf_agent._q_network.summary()

for _ in range(1):
    time_step = eval_env.reset()
    while not time_step.is_last():
        action_step = tf_agent.policy.action(time_step)
        time_step = eval_env.step(action_step.action)
        print(eval_py_env.render())
