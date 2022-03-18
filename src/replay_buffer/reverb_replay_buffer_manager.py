import reverb
from replay_buffer.replay_buffer_manager import ReplayBufferManager
from tf_agents.replay_buffers import reverb_replay_buffer, reverb_utils
from tf_agents.specs import tensor_spec


class ReverbReplayBufferManager(ReplayBufferManager):
    def __init__(self, data_spec, replay_buffer_capacity) -> None:
        super().__init__(data_spec=data_spec)

        self.table_name = "uniform_table"
        self.replay_buffer_capacity = replay_buffer_capacity
        self.replay_buffer_signature = tensor_spec.add_outer_dim(
            self.data_spec)
        self.server = self.create_server()
        self.replay_buffer = self.create_replay_buffer()
        self.observer = self.create_observer()

    def create_server(self):
        table = reverb.Table(
            name=self.table_name,
            max_size=self.replay_buffer_capacity,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            rate_limiter=reverb.rate_limiters.MinSize(10),
            signature=self.replay_buffer_signature
        )

        reverb_server = reverb.Server([table])

        return reverb_server

    def create_replay_buffer(self):
        replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
            data_spec=self.data_spec,
            table_name=self.table_name,
            sequence_length=2,  # ToDo: recheck
            local_server=self.server
        )

        return replay_buffer

    def get_replay_buffer(self):
        return self.replay_buffer

    def create_observer(self):
        observer = reverb_utils.ReverbAddTrajectoryObserver(
            py_client=self.replay_buffer.py_client,
            table_name=self.table_name,
            sequence_length=2  # ToDo: recheck
        )

        return observer

    def get_observer(self):
        return self.observer

    def get_replay_buffer_as_dataset(self, num_parallel_calls, batch_size, num_steps, num_prefetch):
        return self.replay_buffer.as_dataset(sample_batch_size=batch_size, num_steps=num_steps, num_parallel_calls=num_parallel_calls).prefetch(num_prefetch)

    def get_dataset_iterator(self, num_parallel_calls, batch_size, num_steps, num_prefetch):
        dataset = self.get_replay_buffer_as_dataset(
            num_parallel_calls, batch_size, num_steps, num_prefetch)

        return iter(dataset)
