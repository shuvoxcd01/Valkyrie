from agent.agent_factory import AgentFactory
from tf_agents.utils import common
from tf_agents.agents.dqn import dqn_agent


class DqnAgentFactory(AgentFactory):
    def __init__(self, time_step_spec, action_spec, q_network, optimizer,
                 target_update_period, train_step_counter, epsilon_greedy=0.1,
                 n_step_update=1.0, target_update_tau=1.0, td_errors_loss_fn=common.element_wise_squared_loss,
                 gamma=0.99, reward_scale_factor=1.0, gradient_clipping=None, debug_summaries=False,
                 summarize_grads_and_vars=False) -> None:
        super().__init__()

        self.time_step_spec = time_step_spec
        self.action_spec = action_spec
        self.q_network = q_network
        self.optimizer = optimizer
        self.target_update_period = target_update_period
        self.train_step_counter = train_step_counter
        self.epsilon_greedy = epsilon_greedy
        self.n_step_update = n_step_update
        self.target_update_tau = target_update_tau
        self.td_error_loss_fn = td_errors_loss_fn
        self.gamma = gamma
        self.reward_scale_factor = reward_scale_factor
        self.gradient_clipping = gradient_clipping
        self.debug_summaries = debug_summaries
        self.summarize_grads_and_vars = summarize_grads_and_vars

    def get_agent(self):
        agent = dqn_agent.DdqnAgent(
            time_step_spec=self.time_step_spec,
            action_spec=self.action_spec,
            q_network=self.q_network,
            optimizer=self.optimizer,
            epsilon_greedy=self.epsilon_greedy,
            n_step_update=self.n_step_update,
            target_update_tau=self.target_update_tau,
            target_update_period=self.target_update_period,
            td_errors_loss_fn=self.td_error_loss_fn,
            gamma=self.gamma,
            reward_scale_factor=self.reward_scale_factor,
            gradient_clipping=self.gradient_clipping,
            debug_summaries=self.debug_summaries,
            summarize_grads_and_vars=self.summarize_grads_and_vars,
            train_step_counter=self.train_step_counter
        )

        agent.initialize()

        return agent
