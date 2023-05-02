from typing import *

import numpy as np
from mltk.callbacks import Callback, CallbackData, Stage
from torch.utils.tensorboard import SummaryWriter

try:
    # problem: https://github.com/pytorch/pytorch/issues/30966
    import tensorflow as tf
    import tensorboard as tb

    tf.io.gfile = tb.compat.tensorflow_stub.io.gfile
except ImportError:
    pass

__all__ = ['SummaryCallback']


class SummaryCallback(Callback):
    """Callback class that writes metrics to TensorBoard."""

    writer: SummaryWriter
    stage: Optional[Stage]
    stage_stack: List[Stage]
    global_step: int

    def __init__(self, *, summary_dir=None, summary_writer=None, global_step: int = 0):
        if (summary_dir is None) == (summary_writer is None):
            raise ValueError(f'One and only one of `summary_dir` and `summary_writer` should be specified, '
                             f'but not both.')

        if summary_dir is not None:
            summary_writer = SummaryWriter(summary_dir)
        self.writer = summary_writer
        self.stage = None
        self.stage_stack = []
        self.global_step = global_step

    def add_embedding(self, *args, **kwargs):
        kwargs.setdefault('global_step', self.global_step)
        return self.writer.add_embedding(*args, **kwargs)

    def update_metrics(self, metrics):
        if metrics:
            for key, val in metrics.items():
                key = self.stage_stack[-1].type.add_metric_prefix(key)
                if np.shape(val) != ():
                    val = np.mean(val)
                self.writer.add_scalar(key, val, self.global_step)

    def set_global_step(self, step: int):
        self.global_step = step

    def on_stage_begin(self, data: CallbackData):
        self.stage_stack.append(data.stage)

    def on_stage_end(self, data: CallbackData):
        self.stage_stack.pop()

    def on_test_end(self, data: CallbackData):
        self.update_metrics(data.metrics)

    def on_validation_end(self, data: CallbackData):
        self.update_metrics(data.metrics)

    def on_batch_begin(self, data: CallbackData):
        if len(self.stage_stack) == 1:
            self.global_step += 1

    def on_batch_end(self, data: CallbackData):
        if len(self.stage_stack) == 1:
            self.update_metrics(data.metrics)

    def on_epoch_end(self, data: CallbackData):
        self.update_metrics(data.metrics)
