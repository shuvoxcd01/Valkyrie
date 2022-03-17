from abc import ABC, abstractmethod


class ReplayBufferManager(ABC):
    @abstractmethod
    def get_replay_buffer(self):
        pass

    @abstractmethod
    def get_observer(self):
        pass

    @abstractmethod
    def get_replay_buffer_as_dataset(self, num_parallel_calls, batch_size, num_steps, num_prefetch):
        pass

    @abstractmethod
    def get_dataset_iterator(self, num_parallel_calls, batch_size, num_steps, num_prefetch):
        pass
