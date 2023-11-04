from abc import ABC, abstractmethod


class UnifiedReplayBufferManager(ABC):
    def __init__(self, data_spec) -> None:
        self.data_spec = data_spec

    @abstractmethod
    def get_replay_buffer(self):
        pass

    @abstractmethod
    def get_observer(self):
        pass

    @abstractmethod
    def get_replay_buffer_as_dataset(
        self, num_parallel_calls, batch_size, num_steps, num_prefetch
    ):
        pass

    @abstractmethod
    def get_dataset_iterator(
        self, num_parallel_calls, batch_size, num_steps, num_prefetch
    ):
        pass
