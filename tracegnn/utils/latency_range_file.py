import os
from typing import *

import yaml

__all__ = ['TraceGraphLatencyRangeFile']

LATENCY_RANGE_FILE = 'latency_range.yml'


class  TraceGraphLatencyRangeFile(object):
    __slots__ = ['root_dir', 'yaml_path', 'latency_data']

    root_dir: str
    yaml_path: str
    latency_data: Dict[int, Dict[str, float]]

    def __init__(self, root_dir: str, require_exists: bool = False):
        self.root_dir = os.path.abspath(root_dir)
        self.yaml_path = os.path.join(self.root_dir, LATENCY_RANGE_FILE)
        self.latency_data = {}
        if os.path.exists(self.yaml_path):
            with open(self.yaml_path, 'r', encoding='utf-8') as f:
                obj = yaml.safe_load(f.read())
            self.latency_data = {
                int(op_id): v
                for op_id, v in obj.items()
            }
        elif require_exists:
            raise IOError(f'LatencyRangeFile does not exist: {self.yaml_path}')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()

    def __contains__(self, item):
        return int(item) in self.latency_data

    def __getitem__(self, operation_id: int) -> Tuple[float, float]:
        v = self.latency_data[int(operation_id)]
        return v['mean'], v['std']

    def __setitem__(self,
                    operation_id: int,
                    value: Union[Tuple[float, float], Dict[str, float]]):
        self.update_item(operation_id, value)

    def get_item(self, operation_id: int):
        return self.latency_data[int(operation_id)]

    def update_item(self,
                    operation_id: int,
                    value: Union[Tuple[float, float], Dict[str, float]]
                    ):
        if isinstance(value, (tuple, list)) and len(value) == 2:
            mean, std = value
            value = {'mean': mean, 'std': std}

        key = int(operation_id)
        if key not in self.latency_data:
            self.latency_data[key] = {}
        self.latency_data[key].update({k: float(v) for k, v in value.items()})

    def clear(self):
        self.latency_data.clear()

    def flush(self):
        self.dump_to(self.root_dir)

    def dump_to(self, output_dir: str):
        payload = {
            k: v
            for k, v in self.latency_data.items()
        }
        cnt = yaml.safe_dump(payload)
        path = os.path.join(output_dir, LATENCY_RANGE_FILE)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(cnt)
