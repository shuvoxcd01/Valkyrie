from abc import ABC, abstractmethod


class ReplayBufferManager(ABC):
    def __init__(self, data_spec) -> None:
        self.data_spec = data_spec

    @abstractmethod
    def get_replay_buffer(self, name: str):
        pass

    @abstractmethod
    def get_observer(self, name: str):
        pass

    @abstractmethod
    def get_replay_buffer_as_dataset(
        self,
        table_name: str,
        num_parallel_calls: int,
        batch_size: int,
        num_steps: int,
        num_prefetch: int,
    ):
        pass

    @abstractmethod
    def get_dataset_iterator(
        self,
        table_name: str,
        num_parallel_calls: int,
        batch_size: int,
        num_steps: int,
        num_prefetch: int,
    ):
        pass

    @abstractmethod
    def get_observations_as_dataset(
        self,
        table_name: str,
        num_parallel_calls: int,
        batch_size: int,
        num_steps: int,
        num_prefetch: int,
    ):
        pass

    @abstractmethod
    def get_actions_as_dataset(
        self,
        table_name: str,
        num_parallel_calls: int,
        batch_size: int,
        num_steps: int,
        num_prefetch: int,
    ):
        pass

    @abstractmethod
    def get_discounts_as_dataset(
        self,
        table_name: str,
        num_parallel_calls: int,
        batch_size: int,
        num_steps: int,
        num_prefetch: int,
    ):
        pass

    @abstractmethod
    def get_rewards_as_dataset(
        self,
        table_name: str,
        num_parallel_calls: int,
        batch_size: int,
        num_steps: int,
        num_prefetch: int,
    ):
        pass

    @abstractmethod
    def update_keep_only(self, tables_to_keep: str):
        pass
