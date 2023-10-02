import IPython
import imageio
import base64

import tensorflow as tf

from tf_agents.environments import tf_py_environment
from Valkyrie.src.agent.tf_agent.ddqn_agent_factory import DdqnAgentFactory
from Valkyrie.src.checkpoint_manager.agent_checkpoint_manager_factory import AgentCheckpointManagerFactory

from Valkyrie.src.environment.pong_factory import PongFactory
from Valkyrie.src.network.atari_q_network_factory import AtariQNetworkFactory


def embed_mp4(filename):
    """Embeds an mp4 file in the notebook."""
    video = open(filename, 'rb').read()
    b64 = base64.b64encode(video)
    tag = '''
  <video width="640" height="480" controls>
    <source src="data:video/mp4;base64,{0}" type="video/mp4">
  Your browser does not support the video tag.
  </video>'''.format(b64.decode())

    return IPython.display.HTML(tag)


def create_policy_eval_video(policy, eval_env, eval_py_env, filename, num_episodes=5, fps=30):
    filename = filename + ".mp4"
    with imageio.get_writer(filename, fps=fps) as video:
        for _ in range(num_episodes):
            time_step = eval_env.reset()
            video.append_data(eval_py_env.render())
            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = eval_env.step(action_step.action)
                video.append_data(eval_py_env.render())
    return embed_mp4(filename)


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

create_policy_eval_video(policy=tf_agent.policy, eval_env=eval_env,
                         eval_py_env=eval_py_env, filename="agent_performance")
