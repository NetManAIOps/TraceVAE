import json
import math
import random
import shutil
import traceback
from enum import Enum
from functools import wraps
from typing import *

import os
import sys
import mltk
import tensorkit as tk
import numpy as np
import torch
import click
from tensorkit import tensor as T
from tensorkit.examples import utils
from tensorkit.train import Checkpoint

from tracegnn.data import *
from tracegnn.models.trace_vae.evaluation import *
from tracegnn.models.trace_vae.graph_utils import *
from tracegnn.models.trace_vae.tensor_utils import *
from tracegnn.models.trace_vae.types import *
from tracegnn.models.trace_vae.model import *
from tracegnn.models.trace_vae.dataset import *
from tracegnn.utils import *


class NANLossError(Exception):

    def __init__(self, epoch):
        super().__init__(epoch)

    @property
    def epoch(self) -> Optional[int]:
        return self.args[0]

    def __str__(self):
        return f'NaN loss encountered at epoch {self.epoch}'


class OptimizerType(str, Enum):
    ADAM = 'adam'
    RMSPROP = 'rmsprop'


class ExpConfig(mltk.Config):
    model: TraceVAEConfig = TraceVAEConfig()
    device: Optional[str] = 'cpu'
    seed: Optional[int] = 0

    class train(mltk.Config):
        max_epoch: int = 60
        struct_pretrain_epochs: Optional[int] = 40  # number of epochs to pre-train the struct_vae
        ckpt_epoch_freq: Optional[int] = 5
        test_epoch_freq: Optional[int] = 5
        latency_hist_epoch_freq: Optional[int] = 10
        latency_std_hist_epoch_freq: Optional[int] = 5

        use_early_stopping: bool = False
        val_epoch_freq: Optional[int] = 2

        kl_beta: float = 1.0
        warm_up_epochs: Optional[int] = None  # number of epochs to warm-up the prior (KLD)

        l2_reg: float = 0.0001
        z_unit_ball_reg: Optional[float] = None
        z2_unit_ball_reg: Optional[float] = None

        init_batch_size: int = 64
        batch_size: int = 64
        val_batch_size: int = 64

        optimizer: OptimizerType = OptimizerType.RMSPROP
        initial_lr: float = 0.001
        lr_anneal_ratio: float = 0.1
        lr_anneal_epochs: int = 30
        clip_norm: Optional[float] = None
        global_clip_norm: Optional[float] = 10  # important for numerical stability

        test_n_z: int = 10
        num_plot_samples: int = 20

    class test(mltk.Config):
        batch_size: int = 64
        eval_n_z: int = 10
        use_biased: bool = True
        latency_log_prob_weight: bool = True
        clip_nll: Optional[float] = 100_000

    class report(mltk.Config):
        html_ext: str = '.html.gz'

    class dataset(mltk.Config):
        root_dir: str = os.path.abspath('./data/processed')


def main(exp: mltk.Experiment[ExpConfig]):
    # config
    config = exp.config

    # set random seed to encourage reproducibility (does it really work?)
    if config.seed is not None:
        T.random.set_deterministic(True)
        T.random.seed(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)

    # Load data
    id_manager = TraceGraphIDManager(os.path.join(config.dataset.root_dir, 'id_manager'))
    latency_range = TraceGraphLatencyRangeFile(os.path.join(config.dataset.root_dir, 'id_manager'))

    train_db = TraceGraphDB(BytesSqliteDB(os.path.join(config.dataset.root_dir, 'processed', 'train')))
    val_db = TraceGraphDB(BytesSqliteDB(os.path.join(config.dataset.root_dir, 'processed', 'val')))
    test_db = TraceGraphDB(
        BytesMultiDB(
            BytesSqliteDB(os.path.join(config.dataset.root_dir, 'processed', 'test')),
            BytesSqliteDB(os.path.join(config.dataset.root_dir, 'processed', 'test-drop')),
            BytesSqliteDB(os.path.join(config.dataset.root_dir, 'processed', 'test-latency')),
        )
    )
    train_stream = TraceGraphDataStream(
        train_db, id_manager=id_manager, batch_size=config.train.batch_size,
        shuffle=True, skip_incomplete=False,
    )
    val_stream = TraceGraphDataStream(
        val_db, id_manager=id_manager, batch_size=config.train.val_batch_size,
        shuffle=False, skip_incomplete=False,
    )
    test_stream = TraceGraphDataStream(
        test_db, id_manager=id_manager, batch_size=config.test.batch_size,
        shuffle=False, skip_incomplete=False,
    )

    utils.print_experiment_summary(
        exp,
        train_data=train_stream,
        val_data=val_stream,
        test_data=test_stream
    )
    print('Train Data:', train_db)
    print('Val Data:', val_db)
    print('Test Data:', test_db)

    # build the network
    vae: TraceVAE = TraceVAE(
        config.model,
        id_manager.num_operations,
    )
    vae = vae.to(T.current_device())
    params, param_names = utils.get_params_and_names(vae)
    utils.print_parameters_summary(params, param_names)
    print('')
    mltk.print_with_time('Network constructed.')

    # define the training method for a certain model part
    def train_part(params, start_epoch, max_epoch, latency_only, do_final_eval):
        # util to ensure all installed hooks will only run within this context
        in_context = [True]

        def F(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                if in_context[0]:
                    return func(*args, **kwargs)
            return wrapper

        # the train procedure
        try:
            # buffer to collect stds of each p(latency|z)
            latency_std = {}
            for key in ('train', 'val', 'test_normal', 'test_drop', 'test_latency'):
                latency_std[key] = ArrayBuffer(81920)

            def should_collect_latency_std():
                return (
                    config.train.latency_std_hist_epoch_freq and
                    loop.epoch % config.train.latency_std_hist_epoch_freq == 0
                )

            def clear_std_buf():
                for buf in latency_std.values():
                    buf.clear()

            # the initialization function
            def initialize():
                G = TraceGraphBatch(
                    id_manager=id_manager,
                    latency_range=latency_range,
                    trace_graphs=train_db.sample_n(config.train.init_batch_size),
                )
                chain = vae.q(G).chain(
                    vae.p,
                    G=G,
                )
                loss = chain.vi.training.sgvb(reduction='mean')
                mltk.print_with_time(f'Network initialized: loss = {T.to_numpy(loss)}')

            # the train functions
            def on_train_epoch_begin():
                # set train mode
                if latency_only:
                    tk.layers.set_eval_mode(vae)
                    tk.layers.set_train_mode(vae.latency_vae)
                else:
                    tk.layers.set_train_mode(vae)

                # clear std buffer
                clear_std_buf()

            def train_step(trace_graphs):
                G = TraceGraphBatch(
                    id_manager=id_manager,
                    latency_range=latency_range,
                    trace_graphs=trace_graphs,
                )
                chain = vae.q(G).chain(
                    vae.p,
                    G=G,
                )

                # collect the latency std
                if should_collect_latency_std():
                    collect_latency_std(latency_std['train'], chain)

                # collect the log likelihoods
                p_obs = []
                p_latent = []
                q_latent = []
                for name in chain.p:
                    if name in chain.q:
                        q_latent.append(chain.q[name].log_prob())
                        p_latent.append(chain.p[name].log_prob())
                    else:
                        # print(name, chain.p[name].log_prob().mean())
                        p_obs.append(chain.p[name].log_prob())

                # get E[log p(x|z)] and KLD[q(z|x)||p(z)]
                recons = T.reduce_mean(T.add_n(p_obs))
                kl = T.reduce_mean(T.add_n(q_latent) - T.add_n(p_latent))

                # KL beta
                beta = config.train.kl_beta
                if config.train.warm_up_epochs and loop.epoch < config.train.warm_up_epochs:
                    beta = beta * (loop.epoch / config.train.warm_up_epochs)
                loss = beta * kl - recons

                # l2 regularization
                if config.train.l2_reg:
                    l2_params = []
                    for p, n in zip(params, param_names):
                        if 'bias' not in n:
                            l2_params.append(p)
                    loss = loss + config.train.l2_reg * T.nn.l2_regularization(l2_params)

                # unit ball regularization
                def add_unit_ball_reg(l, t, reg):
                    if reg is not None:
                        ball_mean, ball_var = get_moments(t, axis=[-1])
                        l = l + reg * (
                            T.reduce_mean(ball_mean ** 2) +
                            T.reduce_mean((ball_var - 1) ** 2)
                        )
                    return l

                loss = add_unit_ball_reg(loss, chain.q['z'].tensor, config.train.z_unit_ball_reg)
                if 'z2' in chain.q:
                    loss = add_unit_ball_reg(loss, chain.q['z2'].tensor, config.train.z2_unit_ball_reg)

                # check and return the metrics
                loss_val = T.to_numpy(loss)
                if math.isnan(loss_val):
                    raise NANLossError(loop.epoch)

                return {'loss': loss, 'recons': recons, 'kl': kl}

            # the validation function
            def validate():
                tk.layers.set_eval_mode(vae)

                def val_step(trace_graphs):
                    with T.no_grad():
                        G = TraceGraphBatch(
                            id_manager=id_manager,
                            latency_range=latency_range,
                            trace_graphs=trace_graphs,
                        )
                        chain = vae.q(G).chain(
                            vae.p,
                            G=G,
                        )
                        # collect the latency std
                        if should_collect_latency_std():
                            collect_latency_std(latency_std['val'], chain)
                        loss = chain.vi.training.sgvb()
                        return {'loss': T.to_numpy(T.reduce_mean(loss))}

                val_loop = loop.validation()
                result_dict = val_loop.run(val_step, val_stream)
                result_dict = {
                    f'val_{k}': v
                    for k, v in result_dict.items()
                }
                summary_cb.update_metrics(result_dict)

            # the evaluation function
            def evaluate(n_z, eval_loop, eval_stream, epoch, use_embeddings=False,
                         plot_latency_hist=False):
                # latency_hist_file
                latency_hist_file = None
                if plot_latency_hist:
                    latency_hist_file = exp.make_parent(f'./plotting/latency-sample/{epoch}.jpg')

                # do evaluation
                tk.layers.set_eval_mode(vae)
                with T.no_grad():
                    kw = {}
                    if should_collect_latency_std():
                        kw['latency_std_dict_out'] = latency_std
                        kw['latency_dict_prefix'] = 'test_'
                    result_dict = do_evaluate_nll(
                        test_stream=eval_stream,
                        vae=vae,
                        id_manager=id_manager,
                        latency_range=latency_range,
                        n_z=n_z,
                        use_biased=config.test.use_biased,
                        latency_log_prob_weight=config.test.latency_log_prob_weight,
                        test_loop=eval_loop,
                        summary_writer=summary_cb,
                        clip_nll=config.test.clip_nll,
                        use_embeddings=use_embeddings,
                        latency_hist_file=latency_hist_file,
                        **kw,
                    )

                with open(exp.make_parent(f'./result/test-anomaly/{epoch}.json'), 'w', encoding='utf-8') as f:
                    f.write(json.dumps(result_dict))
                eval_loop.add_metrics(**result_dict)

            def save_model(epoch=None):
                epoch = epoch or loop.epoch
                torch.save(vae.state_dict(), exp.make_parent(f'models/{epoch}.pt'))

            # final evaluation
            if do_final_eval:
                tk.layers.set_eval_mode(vae)

                # save the final model
                save_model('final')

                clear_std_buf()
                evaluate(
                    n_z=config.test.eval_n_z,
                    eval_loop=mltk.TestLoop(),
                    eval_stream=test_stream,
                    epoch='final',
                    use_embeddings=True,
                    plot_latency_hist=True,
                )

            else:
                # set train mode at the beginning of each epoch
                loop.on_epoch_begin.do(F(on_train_epoch_begin))

                # the optimizer and learning rate scheduler
                if config.train.optimizer == OptimizerType.ADAM:
                    optimizer = tk.optim.Adam(params)
                elif config.train.optimizer == OptimizerType.RMSPROP:
                    optimizer = tk.optim.RMSprop(params)

                def update_lr():
                    n_cycles = int(
                        loop.epoch //  # (loop.epoch - start_epoch) //
                        config.train.lr_anneal_epochs
                    )
                    lr_discount = config.train.lr_anneal_ratio ** n_cycles
                    optimizer.set_lr(config.train.initial_lr * lr_discount)

                update_lr()
                loop.on_epoch_end.do(F(update_lr))

                # install the validation function and early-stopping
                if config.train.val_epoch_freq:
                    loop.run_after_every(
                        F(validate),
                        epochs=config.train.val_epoch_freq,
                    )

                # install the evaluation function during training
                if config.train.test_epoch_freq:
                    loop.run_after_every(
                        F(lambda: evaluate(
                            n_z=config.train.test_n_z,
                            eval_loop=loop.test(),
                            eval_stream=test_stream,
                            epoch=loop.epoch,
                            plot_latency_hist=(
                                config.train.latency_hist_epoch_freq and
                                loop.epoch % config.train.latency_hist_epoch_freq == 0
                            )
                        )),
                        epochs=config.train.test_epoch_freq,
                    )

                # install the plot and sample functions during training
                def after_epoch():
                    save_model()
                loop.run_after_every(F(after_epoch), epochs=1)

                # train the model
                tk.layers.set_eval_mode(vae)
                on_train_epoch_begin()
                initialize()
                utils.fit_model(
                    loop=loop,
                    optimizer=optimizer,
                    fn=train_step,
                    stream=train_stream,
                    clip_norm=config.train.clip_norm,
                    global_clip_norm=config.train.global_clip_norm,
                    # pass to `loop.run()`
                    limit=max_epoch,
                )
        finally:
            in_context = [False]

    # the train loop
    loop = mltk.TrainLoop(max_epoch=config.train.max_epoch)

    # checkpoint
    ckpt = Checkpoint(vae=vae)
    loop.add_callback(mltk.callbacks.AutoCheckpoint(
        ckpt,
        root_dir=exp.make_dirs('./checkpoint'),
        epoch_freq=config.train.ckpt_epoch_freq,
        max_checkpoints_to_keep=10,
    ))

    # early-stopping
    if config.train.val_epoch_freq and config.train.use_early_stopping:
        loop.add_callback(mltk.callbacks.EarlyStopping(
            checkpoint=ckpt,
            root_dir=exp.abspath('./early-stopping'),
            metric_name='val_loss',
        ))

    # the summary writer
    summary_cb = SummaryCallback(summary_dir=exp.abspath('./summary'))
    loop.add_callback(summary_cb)

    # pre-train the struct_vae
    try:
        with loop:
            start_epoch = 1
            part_params = params
            latency_only = False

            if (config.model.arch == TraceVAEArch.DEFAULT) and config.train.struct_pretrain_epochs:
                # train struct_vae first
                print(f'Start to train vae with {len(part_params)} params ...')
                train_part(
                    list(part_params),
                    start_epoch=start_epoch,
                    max_epoch=config.train.struct_pretrain_epochs,
                    latency_only=latency_only,
                    do_final_eval=False,
                )

                # train latency_vae next
                part_params = [
                    p for n, p in zip(param_names, params)
                    if n.startswith('latency_vae')
                ]
                start_epoch = config.train.struct_pretrain_epochs + 1
                latency_only = True
                print(f'Start to train latency_vae with {len(part_params)} params ...')

            train_part(
                part_params,
                start_epoch=start_epoch,
                max_epoch=config.train.max_epoch,
                latency_only=latency_only,
                do_final_eval=False,
            )

        # do final evaluation
        train_part(
            [],
            start_epoch=-1,
            max_epoch=-1,
            latency_only=False,
            do_final_eval=True,
        )

    except KeyboardInterrupt:
        print(
            'Train interrupted, press Ctrl+C again to skip the final test ...',
            file=sys.stderr,
        )


if __name__ == '__main__':
    with mltk.Experiment(ExpConfig) as exp:
        config = exp.config
        device = config.device or T.first_gpu_device()
        with T.use_device(device):
            retrial = 0
            while True:
                try:
                    main(exp)
                except NANLossError as ex:
                    if ex.epoch != 1 or retrial >= 10:
                        raise
                    retrial += 1
                    print(
                        f'\n'
                        f'Restart the experiment for the {retrial}-th time '
                        f'due to NaN loss at epoch {ex.epoch}.\n',
                        file=sys.stderr
                    )
                    if ex.epoch == 1:
                        for name in ['checkpoint', 'early-stopping', 'models',
                                     'plotting', 'summary']:
                            path = exp.abspath(name)
                            if os.path.isdir(name):
                                shutil.rmtree(path)
                else:
                    break
