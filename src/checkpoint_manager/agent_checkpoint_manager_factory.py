from tf_agents.agents import TFAgent

from checkpoint_manager.agent_checkpoint_manager import AgentCheckpointManager


class AgentCheckpointManagerFactory:
    def __init__(self, base_ckpt_dir: str) -> None:
        self.base_ckpt_dir = base_ckpt_dir

    def get_agent_checkpoint_manager(self, agent: TFAgent):
        return AgentCheckpointManager(base_ckpt_dir=self.base_ckpt_dir, agent=agent)
