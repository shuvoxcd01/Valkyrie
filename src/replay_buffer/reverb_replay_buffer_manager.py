from typing import List, Optional
import reverb
from replay_buffer.replay_buffer_manager import ReplayBufferManager
from tf_agents.replay_buffers import reverb_replay_buffer, reverb_utils
from tf_agents.specs import tensor_spec
import logging


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

        self.logger = logging.getLogger(__file__)

        self._initialize()

    def _initialize(self):
        self.logger.info("Initializing replay buffers.")

        if self.table_names:
            self.servers_map = self.create_servers(self.table_names)
            self.replay_buffers_map = self.create_replay_buffers(self.table_names)
            self.observers_map = self.create_observers(self.table_names)

        self.logger.info(f"Servers map: {self.servers_map}")
        self.logger.info(f"Replay buffers map: {self.replay_buffers_map}")
        self.logger.info(f"Observers map: {self.observers_map}")
        self.logger.info(f"Replay buffers initialized.")

    def add_server(self, table_name):
        self.logger.info(f"Adding server for table: {table_name}")

        table = self.create_table(table_name)
        self.table_names.append(table_name)
        reverb_server = reverb.Server([table])
        self.servers_map[table_name] = reverb_server

        self.logger.info(f"Replay buffer server added: {reverb_server}")
        self.logger.info(f"Servers map: {self.servers_map}")

        return reverb_server

    def add_replay_buffer(self, table_name):
        self.logger.info(f"Adding replay buffer for table: {table_name}")

        replay_buffer = self.create_replay_buffer(table_name)
        self.replay_buffers_map[table_name] = replay_buffer

        self.logger.info("Replay buffer added")
        self.logger.info(f"Replay buffers map: {self.replay_buffers_map}")

        return replay_buffer

    def add_observer(self, table_name):
        self.logger.info(f"Adding observer for table: {table_name}")

        observer = self.create_observer(table_name)
        self.observers_map[table_name] = observer

        self.logger.info("Observer added.")
        self.logger.info(f"Observers map: {self.observers_map}")

        return observer

    def add(self, table_name):
        self.logger.info(
            f"Adding server, replay_buffer and observer for table: {table_name}"
        )

        if not table_name in self.table_names:
            self.logger.info(
                f"Table {table_name} does not exist in registered table names. Creating new server, replay buffer and observer."
            )
            self.add_server(table_name)
            self.add_replay_buffer(table_name)
            self.add_observer(table_name)

        else:
            self.logger.info(f"Table {table_name} aready exists. Nothing added.")

    def destroy(self, table_name):
        self.logger.info(f"Removing table {table_name}")

        if table_name in self.observers_map:
            del self.observers_map[table_name]
            self.logger.debug(f"Removed table {table_name} from observers map.")

        if table_name in self.replay_buffers_map:
            del self.replay_buffers_map[table_name]
            self.logger.debug(f"Removed table {table_name} from replay buffers map.")

        if table_name in self.servers_map:
            server = self.servers_map.pop(table_name)
            server.stop()
            self.logger.debug(f"Server {server} stopped.")
            del server
            self.logger.debug(f"Removed table {table_name} from servers map.")

        if table_name in self.table_names:
            self.table_names.remove(table_name)
            self.logger.debug(f"Removed table {table_name} from table names.")

    def create_servers(self, table_names):
        self.logger.info(f"Creating servers for tables: {table_names}.")
        servers_map = {}

        for name in table_names:
            table = self.create_table(name)
            self.logger.debug(f"Table {name} created.")
            reverb_server = reverb.Server([table])
            servers_map[name] = reverb_server
            self.logger.info(f"Server for table {name} created.")

        return servers_map

    def create_table(self, table_name):
        self.logger.debug(f"Creating table: {table_name}.")
        table = reverb.Table(
            name=table_name,
            max_size=self.replay_buffer_capacity,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            rate_limiter=reverb.rate_limiters.MinSize(10),
            signature=self.replay_buffer_signature,
        )
        self.logger.debug(f"Table {table_name} created.")
        return table

    def create_replay_buffer(self, table_name):
        self.logger.info(f"Creating replay buffer for table: {table_name}.")
        replay_buffer = reverb_replay_buffer.ReverbReplayBuffer(
            data_spec=self.data_spec,
            table_name=table_name,
            sequence_length=2,  # ToDo: recheck
            local_server=self.servers_map[table_name],
        )
        self.logger.info(f"Replay buffer for table {table_name} created.")
        return replay_buffer

    def create_replay_buffers(self, table_names):
        self.logger.info(f"Creating replay buffers for tables: {table_names}.")

        replay_buffers_map = {}

        for name in table_names:
            replay_buffers_map[name] = self.create_replay_buffer(name)

        self.logger.info("Replay buffers created.")

        return replay_buffers_map

    def get_replay_buffer(self, table_name):
        self.logger.info(f"Getting replay buffer for table {table_name}.")
        return self.replay_buffers_map[table_name]

    def create_observer(self, table_name):
        self.logger.info(f"Creating observer for table {table_name}.")
        replay_buffer = self.replay_buffers_map[table_name]
        observer = reverb_utils.ReverbAddTrajectoryObserver(
            py_client=replay_buffer.py_client,
            table_name=table_name,
            sequence_length=2,  # ToDo: recheck
        )
        self.logger.info(f"Observer for table {table_name} created.")
        return observer

    def create_observers(self, table_names):
        observers_map = {}

        for name in table_names:
            observers_map[name] = self.create_observer(name)

        return observers_map

    def get_observer(self, table_name: str):
        if table_name not in self.table_names:
            self.logger.info(f"Table  name {table_name} not found")
            self.add(table_name)

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

    def get_observation_iterator(
        self,
        table_name: str,
        num_parallel_calls: Optional[int] = None,
        batch_size: Optional[int] = None,
        num_steps: Optional[int] = None,
        num_prefetch: Optional[int] = None,
    ):
        observation_dataset = self.get_observations_as_dataset(
            table_name=table_name,
            num_parallel_calls=num_parallel_calls,
            batch_size=batch_size,
            num_steps=num_steps,
            num_prefetch=num_prefetch,
        )

        return iter(observation_dataset)

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

        self.logger.info(f"List of tables after updating: {self.table_names}")
        for table in self.table_names:
            self.logger.info(
                f"Table {table} num frames: {self.get_replay_buffer(table).num_frames()}"
            )
