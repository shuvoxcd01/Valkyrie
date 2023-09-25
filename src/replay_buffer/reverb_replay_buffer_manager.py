from typing import List, Optional
import reverb
from replay_buffer.replay_buffer_manager import ReplayBufferManager
from tf_agents.replay_buffers import reverb_replay_buffer, reverb_utils
from tf_agents.specs import tensor_spec


class ReverbReplayBufferManager(ReplayBufferManager):
    def __init__(
        self,
        data_spec,
        replay_buffer_capacity,
        num_parallel_calls: int,
        batch_size: int,
        num_steps: int,
        num_prefetch: int,
        table_names: Optional[List[str]] = None,
    ) -> None:
        super().__init__(data_spec=data_spec)

        self.table_names = table_names if table_names is not None else []
        self.replay_buffer_capacity = replay_buffer_capacity
        self.replay_buffer_signature = tensor_spec.add_outer_dim(self.data_spec)

        self.num_parallel_calls = num_parallel_calls
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.num_prefetch = num_prefetch

        self.servers_map = {}
        self.replay_buffers_map = {}
        self.observers_map = {}

        self._initialize()

    def _initialize(self):
        if self.table_names:
            self.servers_map = self.create_servers(self.table_names)
            self.replay_buffers_map = self.create_replay_buffers(self.table_names)
            self.observers_map = self.create_observers(self.table_names)

    def add_server(self, table_name):
        table = self.create_table(table_name)
        self.table_names.append(table_name)
        reverb_server = reverb.Server([table])
        self.servers_map[table_name] = reverb_server

        return reverb_server

    def add_replay_buffer(self, table_name):
        replay_buffer = self.create_replay_buffer(table_name)
        self.replay_buffers_map[table_name] = replay_buffer

        return replay_buffer

    def add_observer(self, table_name):
        observer = self.create_observer(table_name)
        self.observers_map[table_name] = observer

        return observer

    def add(self, table_name):
        if not table_name in self.table_names:
            self.add_server(table_name)
            self.add_replay_buffer(table_name)
            self.add_observer(table_name)

    def destroy(self, table_name):
        if table_name in self.observers_map:
            del self.observers_map[table_name]

        if table_name in self.replay_buffers_map:
            del self.replay_buffers_map[table_name]

        if table_name in self.servers_map:
            server = self.servers_map.pop(table_name)
            server.stop()
            self.table_names.remove(table_name)

    def create_servers(self, table_names):
        servers_map = {}

        for name in table_names:
            table = self.create_table(name)
            reverb_server = reverb.Server([table])
            servers_map[name] = reverb_server

        return servers_map

    def create_table(self, table_name):
        table = reverb.Table(
            name=table_name,
            max_size=self.replay_buffer_capacity,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            rate_limiter=reverb.rate_limiters.MinSize(10),
            signature=self.replay_buffer_signature,
        )

        return table

    def create_replay_buffer(self, table_name):
        replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
            data_spec=self.data_spec,
            table_name=table_name,
            sequence_length=2,  # ToDo: recheck
            local_server=self.servers_map[table_name],
        )

        return replay_buffer

    def create_replay_buffers(self, table_names):
        replay_buffers_map = {}

        for name in table_names:
            replay_buffers_map[name] = self.create_replay_buffer(name)

        return replay_buffers_map

    def get_replay_buffer(self, table_name):
        return self.replay_buffers_map[table_name]

    def create_observer(self, table_name):
        replay_buffer = self.replay_buffers_map[table_name]
        observer = reverb_utils.ReverbAddTrajectoryObserver(
            py_client=replay_buffer.py_client,
            table_name=table_name,
            sequence_length=2,  # ToDo: recheck
        )

        return observer

    def create_observers(self, table_names):
        observers_map = {}

        for name in table_names:
            observers_map[name] = self.create_observer(name)

        return observers_map

    def get_observer(self, table_name: str):
        return self.observers_map[table_name]

    def get_replay_buffer_as_dataset(
        self,
        table_name: str,
        num_parallel_calls: Optional[int] = None,
        batch_size: Optional[int] = None,
        num_steps: Optional[int] = None,
        num_prefetch: Optional[int] = None,
    ):
        num_parallel_calls = (
            num_parallel_calls
            if num_parallel_calls is not None
            else self.num_parallel_calls
        )
        batch_size = batch_size if batch_size is not None else self.batch_size
        num_steps = num_steps if num_steps is not None else self.num_steps
        num_prefetch = num_prefetch if num_prefetch is not None else self.num_prefetch

        replay_buffer = self.replay_buffers_map[table_name]

        return replay_buffer.as_dataset(
            sample_batch_size=batch_size,
            num_steps=num_steps,
            num_parallel_calls=num_parallel_calls,
        ).prefetch(num_prefetch)

    def get_dataset_iterator(
        self,
        table_name: str,
        num_parallel_calls: Optional[int] = None,
        batch_size: Optional[int] = None,
        num_steps: Optional[int] = None,
        num_prefetch: Optional[int] = None,
    ):
        dataset = self.get_replay_buffer_as_dataset(
            table_name, num_parallel_calls, batch_size, num_steps, num_prefetch
        )

        return iter(dataset)

    def get_observations_as_dataset(
        self,
        table_name: str,
        num_parallel_calls: Optional[int] = None,
        batch_size: Optional[int] = None,
        num_steps: Optional[int] = None,
        num_prefetch: Optional[int] = None,
    ):
        dataset = self.get_replay_buffer_as_dataset(
            table_name=table_name,
            num_parallel_calls=num_parallel_calls,
            batch_size=batch_size,
            num_steps=num_steps,
            num_prefetch=num_prefetch,
        )

        return dataset.map(lambda trajectory, info: trajectory.observation)

    def get_actions_as_dataset(
        self,
        table_name: str,
        num_parallel_calls: Optional[int] = None,
        batch_size: Optional[int] = None,
        num_steps: Optional[int] = None,
        num_prefetch: Optional[int] = None,
    ):
        dataset = self.get_replay_buffer_as_dataset(
            table_name=table_name,
            num_parallel_calls=num_parallel_calls,
            batch_size=batch_size,
            num_steps=num_steps,
            num_prefetch=num_prefetch,
        )

        return dataset.map(lambda trajectory, info: trajectory.action)

    def get_discounts_as_dataset(
        self,
        table_name: str,
        num_parallel_calls: Optional[int] = None,
        batch_size: Optional[int] = None,
        num_steps: Optional[int] = None,
        num_prefetch: Optional[int] = None,
    ):
        dataset = self.get_replay_buffer_as_dataset(
            table_name=table_name,
            num_parallel_calls=num_parallel_calls,
            batch_size=batch_size,
            num_steps=num_steps,
            num_prefetch=num_prefetch,
        )

        return dataset.map(lambda trajectory, info: trajectory.discount)

    def get_rewards_as_dataset(
        self,
        table_name: str,
        num_parallel_calls: Optional[int] = None,
        batch_size: Optional[int] = None,
        num_steps: Optional[int] = None,
        num_prefetch: Optional[int] = None,
    ):
        dataset = self.get_replay_buffer_as_dataset(
            table_name=table_name,
            num_parallel_calls=num_parallel_calls,
            batch_size=batch_size,
            num_steps=num_steps,
            num_prefetch=num_prefetch,
        )

        return dataset.map(lambda trajectory, info: trajectory.reward)

    def get_all_observers(self) -> List:
        return list(self.observers_map.values())

    def update_keep_only(self, tables_to_keep: List[str]):
        tables_to_keep.append("PRETRAIN")  # ToDo: Fix

        tables_to_delete = list(set(self.table_names) - set(tables_to_keep))

        for table in tables_to_delete:
            self.destroy(table)

        tables_to_add = list(set(tables_to_keep) - set(self.table_names))

        for table in tables_to_add:
            self.add(table)
