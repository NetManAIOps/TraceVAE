import numpy as np

__all__ = ['ArrayBuffer']


class ArrayBuffer(object):

    __slots__ = ['length', 'capacity', 'dtype', 'buffer']

    def __init__(self, capacity: int = 32, dtype=np.float32):
        self.length = 0
        self.capacity = capacity
        self.dtype = dtype
        self.buffer = np.empty([capacity], dtype=dtype)

    def __len__(self):
        return self.length

    def __iter__(self):
        return iter(self.array)

    @property
    def array(self):
        return self.buffer[:self.length]

    def extend(self, items):
        offset = self.length
        new_length = len(items)
        req_capacity = new_length + offset
        if req_capacity > self.capacity:
            self.capacity = capacity = max(self.capacity * 2, req_capacity)
            buffer = np.empty([capacity], dtype=self.dtype)
            buffer[:offset] = self.buffer[:offset]
            self.buffer = buffer
        self.buffer[offset: offset + new_length] = items
        self.length += new_length

    def clear(self):
        self.length = 0
