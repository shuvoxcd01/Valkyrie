from tf_agents.agents import TFAgent


class MetaAgent:
    def __init__(self, tf_agent: TFAgent) -> None:
        self.tf_agent = tf_agent
        self.previous_fitness = 0.0
        self.fitness = 0.0
        self.tweak_probability = None
