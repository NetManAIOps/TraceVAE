import os

import yaml

__all__ = ['IDAssign']


class IDAssign(object):

    def __init__(self, path: str):
        self._path = path
        self._mapping = {'': 0}  # by default let 0 == '' (a NULL item)

        if os.path.isfile(path):
            with open(path, 'r', encoding='utf-8') as f:
                self._mapping = yaml.safe_load(f.read())

        if self._mapping:
            self._next_index = max(self._mapping.values()) + 1
            self._rev_mapping = {v: k for k, v in self._mapping.items()}
        else:
            self._next_index = 0
            self._rev_mapping = {}

    def __len__(self):
        return self._next_index

    def __getitem__(self, key):
        return self._mapping[key]

    @property
    def path(self) -> str:
        return self._path

    def dump_to(self, path: str):
        cnt = yaml.safe_dump(self._mapping)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(cnt)

    def get_or_assign(self, key: str):
        ret = self._mapping.get(key, None)
        if ret is None:
            self._mapping[key] = ret = self._next_index
            self._rev_mapping[ret] = key
            self._next_index += 1
        return ret

    def reverse_map(self, index: int):
        return self._rev_mapping[index]

    def flush(self):
        self.dump_to(self._path)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()
