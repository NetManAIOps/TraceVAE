"""Wraps a BytesDB into TraceGraphDB."""
import os
import pickle as pkl
import re
from contextlib import contextmanager
from typing import *

import numpy as np

from .bytes_db import *
from .trace_graph import *

__all__ = ['TraceGraphDB', 'open_trace_graph_db']


class TraceGraphDB(object):
    bytes_db: BytesDB
    protocol: int

    def __init__(self, bytes_db: BytesDB, protocol: Optional[int] = None):
        if protocol is None:
            protocol = pkl.DEFAULT_PROTOCOL
        self.bytes_db = bytes_db
        self.protocol = protocol

    def __enter__(self):
        self.bytes_db.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.bytes_db.__exit__(exc_type, exc_val, exc_tb)

    def __len__(self) -> int:
        return self.data_count()

    def __getitem__(self, item: int):
        return self.get(item)

    def __iter__(self):
        for i in range(self.data_count()):
            yield self.get(i)

    def __repr__(self):
        desc = repr(self.bytes_db)
        desc = desc[desc.find('(') + 1: -1]
        return f'TraceGraphDB({desc})'

    def sample_n(self,
                 n: int,
                 with_id: bool = False
                 ) -> List[Union[TraceGraph, Tuple[int, TraceGraph]]]:
        ret = []
        indices = np.random.randint(self.data_count(), size=n)
        for i in indices:
            g = self.get(i)
            if with_id:
                ret.append((int(i), g))
            else:
                ret.append(g)
        return ret

    def data_count(self) -> int:
        return self.bytes_db.data_count()

    def get(self, item: int) -> TraceGraph:
        return TraceGraph.from_bytes(self.bytes_db.get(item))

    def add(self, g: TraceGraph) -> int:
        return self.bytes_db.add(g.to_bytes(protocol=self.protocol))

    @contextmanager
    def write_batch(self):
        with self.bytes_db.write_batch():
            yield self

    def commit(self):
        self.bytes_db.commit()

    def optimize(self):
        self.bytes_db.optimize()

    def close(self):
        self.bytes_db.close()


def open_trace_graph_db(input_dir: str,
                        names: Optional[Sequence[str]] = (),
                        protocol: Optional[int] = None,
                        ) -> Tuple[TraceGraphDB, TraceGraphIDManager]:
    file_name = f'_bytes_{protocol}.db' if protocol else '_bytes.db'

    id_manager = TraceGraphIDManager(os.path.join(input_dir, 'id_manager'))

    if len(names) == 1:
        db = TraceGraphDB(
            BytesSqliteDB(os.path.join(input_dir, 'processed', names[0]), file_name=file_name),
            protocol=protocol,
        )
    else:
        db = TraceGraphDB(
            BytesMultiDB(*[
                BytesSqliteDB(os.path.join(input_dir, 'processed', name), file_name=file_name)
                for name in names
            ]),
            protocol=protocol,
        )

    return db, id_manager
