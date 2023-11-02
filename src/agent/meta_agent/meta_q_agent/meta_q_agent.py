import logging
from tf_agents.agents import TFAgent
from Valkyrie.src.agent.meta_agent.meta_q_agent.meta_q_agent_copier import (
    MetaQAgentCopier,
)
from Valkyrie.src.evolutionary_operations.evo_ops_nn import EvolutionaryOperationsNN
from agent.meta_agent.meta_agent import MetaAgent
from summary_writer.summary_writer_manager import SummaryWriterManager
from checkpoint_manager.agent_checkpoint_manager import AgentCheckpointManager
from Valkyrie.src.agent.meta_agent.meta_agent import MetaAgent
import tensorflow_probability as tfp
from tf_agents.networks import Network, Sequential
import logging


class MetaQAgent(MetaAgent):
    def __init__(
        self,
        tf_agent: TFAgent,
        checkpoint_manager: AgentCheckpointManager,
        summary_writer_manager: SummaryWriterManager,
        agent_copier: MetaQAgentCopier,
        fitness=0,
        previous_fitness=0,
        generation: int = 0,
        name=None,
    ) -> None:
        super().__init__(
            tf_agent=tf_agent,
            checkpoint_manager=checkpoint_manager,
            summary_writer_manager=summary_writer_manager,
            fitness=fitness,
            previous_fitness=previous_fitness,
            generation=generation,
            name=name,
        )
        self.logger = logging.getLogger()
        self.agent_copier = agent_copier

    def mutate(self, mean: float = 0.0, variance: float = 0.0001):
        self.logger.debug(f"Performing mutation: {self.tf_agent.name}")
        q_net = self.get_network()

        layers_to_mutate = q_net.get_mutation_layers()
        EvolutionaryOperationsNN.mutate(
            mutation_layers=layers_to_mutate, mean=mean, variance=variance
        )
        self.logger.debug(f"{self.name}: Mutation performed.")

    def get_network(self):
        return self.tf_agent._q_network

    def crossover(self, partner, self_keep_percentage):
        assert type(self) == type(partner), "Types of crossover partners don't match."

        generation = self.generation
        name = self.name
        parent_1_network = self.get_network()
        parent_2_network = partner.get_network()
        child_network = parent_1_network.copy()

        if isinstance(child_network, Network) and not isinstance(
            child_network, Sequential
        ):
            child_network.create_variables()
            child_network.set_weights(parent_1_network.get_weights())

        child_crossover_layers = child_network.get_crossover_layers()
        parent_1_crossover_layers = parent_1_network.get_crossover_layers()
        parent_2_crossover_layers = parent_2_network.get_crossover_layers()

        EvolutionaryOperationsNN.crossover(
            self_keep_percentage,
            parent_1_layers=parent_1_crossover_layers,
            parent_2_layers=parent_2_crossover_layers,
            child_layers=child_crossover_layers,
        )

        child_meta_agent = self.agent_copier.copy_agent(
            meta_agent=self,
            name=name,
            agent_generation=generation,
            training_step_counter=0,
            network=child_network,
        )

        return child_meta_agent

    def copy(self, name: str = None, generation: int = None):
        if name is None:
            name = self.name

        if generation is None:
            generation = self.generation

        return self.agent_copier.copy_agent(
            meta_agent=self, name=name, agent_generation=generation
        )

    def save(self):
        self.checkpoint_manager.save_checkpointer()

    def delete(self):
        self.checkpoint_manager.delete_checkpointer()
