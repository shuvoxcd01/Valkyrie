from tf_agents.drivers.driver import Driver
from tf_agents.agents import TFAgent
from agent.meta_agent.meta_agent import MetaAgent
from summary_writer.summary_writer_manager import SummaryWriterManager
from checkpoint_manager.agent_checkpoint_manager import AgentCheckpointManager
import tensorflow_probability as tfp


class MetaQAgent(MetaAgent):
    def __init__(self, tf_agent: TFAgent, checkpoint_manager: AgentCheckpointManager,
                 summary_writer_manager: SummaryWriterManager,
                 fitness=0, previous_fitness=0,
                 tweak_probability=0, beta=0, generation: int = 0) -> None:
        super().__init__(tf_agent=tf_agent, checkpoint_manager=checkpoint_manager,
                         summary_writer_manager=summary_writer_manager,
                         fitness=fitness, previous_fitness=previous_fitness,
                         tweak_probability=tweak_probability, beta=beta,
                         generation=generation)

    def mutate(self):
        d = tfp.distributions.Normal(loc=0., scale=0.1)

        for layer in self.tf_agent._q_network.layers:
            if layer.trainable:
                weights_list = layer.get_weights()
                new_weights_list = []
                for weights in weights_list:
                    noise = d.sample(sample_shape=weights.shape)
                    new_weights = weights + noise
                    new_weights_list.append(new_weights)
                layer.set_weights(new_weights_list)
