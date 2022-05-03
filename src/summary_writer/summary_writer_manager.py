import tensorflow as tf
import os
from tf_agents.agents import TFAgent


class SummaryWriterManager:
    def __init__(self, base_summary_writer_dir: str, agent: TFAgent) -> None:
        self.base_dir = base_summary_writer_dir
        self.agent = agent
        self.summary_writer = self._get_summary_writer()

    def _get_summary_writer(self):
        summary_writer_dir = os.path.join(
            self.base_dir, self.agent.name)
        summary_writer = tf.summary.create_file_writer(summary_writer_dir)

        return summary_writer

    def write_scalar_summary(self, name, data, step=None):
        with self.summary_writer.as_default():
            agent_step = self.agent.train_step_counter.numpy() if step is None else step
            tf.summary.scalar(name=name, data=data, step=agent_step)
            self.summary_writer.flush()

    def close_writer(self):
        self.summary_writer.close()
