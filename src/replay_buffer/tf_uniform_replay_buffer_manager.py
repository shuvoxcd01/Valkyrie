from tf_agents.replay_buffers import tf_uniform_replay_buffer

from replay_buffer.replay_buffer_manager import ReplayBufferManager


class TFUniformReplayBufferManager(ReplayBufferManager):
    def __init__(self, data_spec, batch_size, max_length) -> None:
        self.data_spec = data_spec
        self.batch_size = batch_size
        self.max_length = max_length
        self.replay_buffer = self.create_replay_buffer()

    def create_replay_buffer(self):
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self.data_spec,
            batch_size=self.batch_size,
            max_length=self.max_length
        )

        return replay_buffer

    def get_replay_buffer(self):
        return self.replay_buffer

    def get_observer(self):
        observer = self.replay_buffer.add_batch
        return observer

    def get_replay_buffer_as_dataset(self, num_parallel_calls, batch_size, num_steps, num_prefetch):
        return self.replay_buffer.as_dataset(
            num_parallel_calls=num_parallel_calls,
            sample_batch_size=batch_size,
            num_steps=num_steps
        ).prefetch(num_prefetch)

    def get_dataset_iterator(self, num_parallel_calls, batch_size, num_steps, num_prefetch):
        dataset = self.get_replay_buffer_as_dataset(
            num_parallel_calls, batch_size, num_steps, num_prefetch)

        return iter(dataset)
