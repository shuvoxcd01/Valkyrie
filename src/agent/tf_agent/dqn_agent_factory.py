from agent.tf_agent.agent_factory import AgentFactory
from tf_agents.utils import common
from tf_agents.agents.dqn import dqn_agent
from tf_agents.networks import Network
import tensorflow as tf
from tf_agents.typing.types import Optimizer


class DqnAgentFactory(AgentFactory):
    def __init__(self, time_step_spec, action_spec, td_errors_loss_fn=common.element_wise_squared_loss) -> None:
        super().__init__()

        self.time_step_spec = time_step_spec
        self.action_spec = action_spec
        self.td_error_loss_fn = td_errors_loss_fn

    def get_agent(self, name: str, network: Network, optimizer: Optimizer, train_step_counter: tf.Variable):
        agent = dqn_agent.DqnAgent(
            name=name,
            time_step_spec=self.time_step_spec,
            action_spec=self.action_spec,
            q_network=network,
            optimizer=optimizer,
            td_errors_loss_fn=self.td_error_loss_fn,
            train_step_counter=train_step_counter
        )

        agent.initialize()

        return agent
