from tf_agents.replay_buffers import tf_uniform_replay_buffer


class TFUniformReplayBufferManager:
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
