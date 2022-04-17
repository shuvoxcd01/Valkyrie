from summary_writer.summary_writer_manager import SummaryWriterManager
from tf_agents.agents import TFAgent


class SummaryWriterManagerFactory:
    def __init__(self, base_summary_writer_dir: str) -> None:
        self.base_summary_writer_dir = base_summary_writer_dir

    def get_summary_writer_manager(self, tf_agent: TFAgent):
        return SummaryWriterManager(base_summary_writer_dir=self.base_summary_writer_dir, agent=tf_agent)
