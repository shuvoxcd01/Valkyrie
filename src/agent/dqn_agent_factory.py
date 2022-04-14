from agent.agent_factory import AgentFactory
from tf_agents.utils import common
from tf_agents.agents.dqn import dqn_agent


class DqnAgentFactory(AgentFactory):
    def __init__(self, time_step_spec, action_spec, q_network, optimizer, train_step_counter, td_errors_loss_fn=common.element_wise_squared_loss) -> None:
        super().__init__()

        self.time_step_spec = time_step_spec
        self.action_spec = action_spec
        self.q_network = q_network
        self.optimizer = optimizer
        self.train_step_counter = train_step_counter
        self.td_error_loss_fn = td_errors_loss_fn

    def get_agent(self):
        agent = dqn_agent.DqnAgent(
            time_step_spec=self.time_step_spec,
            action_spec=self.action_spec,
            q_network=self.q_network,
            optimizer=self.optimizer,
            td_errors_loss_fn=self.td_error_loss_fn,
            train_step_counter=self.train_step_counter
        )

        agent.initialize()

        return agent
