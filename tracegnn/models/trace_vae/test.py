import pickle
from pprint import pprint
from tempfile import TemporaryDirectory

import os
import mltk
import click
import tensorkit as tk
import numpy as np
from tensorkit import tensor as T
from tensorkit.examples.utils import print_experiment_summary

from tracegnn.data import *
from tracegnn.models.trace_vae.dataset import TraceGraphDataStream
from tracegnn.models.trace_vae.evaluation import *
from tracegnn.models.trace_vae.graph_utils import *
from tracegnn.models.trace_vae.test_utils import *
from tracegnn.models.trace_vae.types import TraceGraphBatch
from tracegnn.utils import *


@click.group()
def main():
    pass


@main.command(context_settings=dict(
    ignore_unknown_options=True,
    help_option_names=[],
))
@click.option('-D', '--data-dir', required=False)
@click.option('-M', '--model-path', required=True)
@click.option('-o', '--nll-out', required=False, default=None)
@click.option('--proba-out', default=None, required=False)
@click.option('--auc-out', default=None, required=False)
@click.option('--latency-out', default=None, required=False)
@click.option('--gui', is_flag=True, default=False, required=False)
@click.option('--device', required=False, default=None)
@click.option('--n_z', type=int, required=False, default=10)
@click.option('--batch-size', type=int, default=128)
@click.option('--clip-nll', type=float, default=100_000)
@click.option('--no-biased', is_flag=True, default=False, required=False)
@click.option('--no-latency-biased', is_flag=True, default=False, required=False)
@click.option('--no-latency', is_flag=True, default=False, required=False)
@click.option('--use-train-val', is_flag=True, default=False, required=False)
@click.option('--infer-bias-std', is_flag=True, default=False, required=False)
@click.option('--bias-std-normal-p', type=float, default=0.995, required=False)
@click.option('--infer-threshold', is_flag=True, default=False, required=False)
@click.option('--threshold-p', type=float, default=0.995, required=False)
@click.option('--threshold-amplify', type=float, default=1.0, required=False)
@click.option('--no-latency-log-prob-weight', is_flag=True, default=False, required=False)
@click.option('--use-std-limit', is_flag=True, default=False, required=False)
@click.option('--std-limit-global', is_flag=True, default=False, required=False)
@click.option('--std-limit-fixed', type=float, default=None, required=False)
@click.option('--std-limit-p', type=float, default=0.99, required=False)
@click.option('--std-limit-amplify', type=float, default=1.0, required=False)
@click.argument('extra_args', nargs=-1, type=click.UNPROCESSED)
def evaluate_nll(data_dir, model_path, nll_out, proba_out, auc_out, latency_out, gui, device,
                 n_z, batch_size, clip_nll, no_biased, no_latency_biased, no_latency,
                 use_train_val, infer_bias_std, bias_std_normal_p, infer_threshold,
                 threshold_p, threshold_amplify, no_latency_log_prob_weight,
                 use_std_limit, std_limit_global, std_limit_fixed, std_limit_p, std_limit_amplify,
                 extra_args):
    N_LIMIT = None

    if infer_bias_std or infer_threshold or use_std_limit:
        use_train_val = True

    with mltk.Experiment(mltk.Config, args=[]) as exp:
        # check parameters
        if gui:
            proba_out = ':show:'
            auc_out = ':show:'
            latency_out = ':show:'

        with T.use_device(device or T.first_gpu_device()):
            # load the config
            train_config = load_config(
                model_path=model_path,
                strict=False,
                extra_args=extra_args,
            )
            if data_dir is None:
                data_dir = train_config.dataset.root_dir

            # load the dataset
            data_names = ['test', 'test-drop', 'test-latency']
            test_db, id_manager = open_trace_graph_db(
                data_dir,
                names=data_names
            )
            print('Test DB:', test_db)
            latency_range = TraceGraphLatencyRangeFile(
                id_manager.root_dir,
                require_exists=True,
            )
            test_stream = TraceGraphDataStream(
                test_db, id_manager=id_manager, batch_size=batch_size,
                shuffle=False, skip_incomplete=False, data_count=N_LIMIT,
            )

            # also load train / val
            if use_train_val:
                train_db, _ = open_trace_graph_db(
                    data_dir,
                    names=['train'],
                )
                print('Train DB:', train_db)
                val_db, _ = open_trace_graph_db(
                    data_dir,
                    names=['val']
                )
                print('Val DB:', val_db)
                train_stream = TraceGraphDataStream(
                    train_db, id_manager=id_manager, batch_size=batch_size,
                    shuffle=True, skip_incomplete=False, data_count=N_LIMIT,
                )
                val_stream = TraceGraphDataStream(
                    val_db, id_manager=id_manager, batch_size=batch_size,
                    shuffle=True, skip_incomplete=False, data_count=N_LIMIT,
                )
            else:
                train_stream = val_stream = None

            print_experiment_summary(exp, train_stream, val_stream, test_stream)

            # load the model
            vae = load_model2(
                model_path=model_path,
                train_config=train_config,
                id_manager=id_manager,
            )
            mltk.print_config(vae.config, title='Model Config')
            vae = vae.to(T.current_device())

            # do evaluation
            operation_id = {}
            latency_std = {}
            latency_reldiff = {}
            p_node_count = {}
            p_edge = {}
            nll_result = {}
            thresholds = {}
            std_group_limit = np.full([id_manager.num_operations], np.nan, dtype=np.float32)

            def F(stream, category, n_z, threshold=None, std_limit=None):
                # the save files kw
                kw = dict(
                    nll_output_file=ensure_parent_exists(nll_out),
                    proba_cdf_file=ensure_parent_exists(proba_out),
                    auc_curve_file=ensure_parent_exists(auc_out),
                    latency_hist_file=ensure_parent_exists(latency_out),
                )
                differ_set = set()

                for k in kw:
                    if kw[k] is not None:
                        s = kw[k].replace('test', category)
                        if category == 'test' or s != kw[k]:
                            differ_set.add(k)
                        kw[k] = s
                kw = {k: v for k, v in kw.items() if k in differ_set}

                # the output temp dir
                with TemporaryDirectory() as temp_dir:
                    if 'nll_output_file' not in kw:
                        kw['nll_output_file'] = ensure_parent_exists(
                            os.path.join(temp_dir, 'nll.npz')
                        )

                    # do evaluation
                    result_dict = do_evaluate_nll(
                        test_stream=stream,
                        vae=vae,
                        id_manager=id_manager,
                        latency_range=latency_range,
                        n_z=n_z,
                        use_biased=(not no_biased) and (category == 'test'),
                        use_latency_biased=not no_latency_biased,
                        no_latency=no_latency,
                        no_struct=False,
                        latency_log_prob_weight=not no_latency_log_prob_weight,
                        std_limit=std_limit,
                        test_threshold=threshold,
                        clip_nll=clip_nll,
                        use_embeddings=False,
                        operation_id_dict_out=operation_id,
                        latency_std_dict_out=latency_std,
                        p_node_count_dict_out=p_node_count,
                        p_edge_dict_out=p_edge,
                        latency_reldiff_dict_out=latency_reldiff,
                        latency_dict_prefix=f'{category}_',
                        **kw,
                    )
                    result_dict = {f'{category}_{k}': v for k, v in result_dict.items()}
                    exp.doc.update({'result': result_dict}, immediately=True)
                    pprint(result_dict)

                    # load the NLLs if category in ('train', 'val')
                    if category in ('train', 'val'):
                        nll_result[category] = np.load(kw['nll_output_file'])['nll_list']

            tk.layers.set_eval_mode(vae)
            with T.no_grad():
                if use_train_val:
                    F(train_stream, 'train', 1)
                    F(val_stream, 'val', 1)

                    if infer_bias_std:
                        bias_std = np.percentile(latency_reldiff['val_normal'].array, bias_std_normal_p * 100)
                        exp.doc.update({'result': {'bias_std': bias_std}}, immediately=True)
                        print(f'Set bias_std = {bias_std:.3f}, bias_std_normal_p = {bias_std_normal_p:.3f}')
                        vae.config.latency.decoder.biased_normal_std_threshold = bias_std

                    if infer_threshold:
                        for category in ('train', 'val'):
                            th_cand = []
                            for _ in range(10):
                                nll_subset = nll_result[category]
                                nll_subset = np.random.choice(nll_subset, replace=True, size=len(nll_subset))
                                if clip_nll:
                                    nll_subset = nll_subset[nll_subset < clip_nll - 1e-7]
                                else:
                                    nll_subset = nll_subset[np.isfinite(nll_subset)]
                                th = np.percentile(nll_subset, threshold_p * 100) * threshold_amplify
                                th_cand.append(th)
                            thresholds[f'{category}_threshold'] = th = np.median(th_cand)
                            print(
                                f'Set {category}_threshold = {th:.3f}, '
                                f'threshold_p = {threshold_p:.3f}, '
                                f'threshold_amplify = {threshold_amplify:.3f}'
                            )
                        exp.doc.update({'result': thresholds}, immediately=True)

                    if use_std_limit:
                        if std_limit_fixed is not None:
                            print(f'Std limit fixed: {std_limit_fixed:.4f}')
                            std_group_limit[:] = std_limit_fixed
                        elif std_limit_global:
                            key = 'val_normal'
                            std_limit = float(np.percentile(
                                latency_std[key].array,
                                std_limit_p * 100
                            ))
                            print(f'Std limit: {std_limit:.4f}')
                            std_group_limit[:] = std_limit
                        else:
                            key = 'val_normal'
                            v1 = operation_id[key].array
                            v2 = latency_std[key].array
                            max_limit = 0

                            for srv_id in range(id_manager.num_operations):
                                v = v2[v1 == srv_id]
                                if len(v) > 0:
                                    srv_limit = (
                                        std_limit_amplify *
                                        float(np.percentile(v, std_limit_p * 100))
                                    )
                                    std_group_limit[srv_id] = srv_limit
                                    max_limit = max(max_limit, srv_limit)

                            for srv_id in range(id_manager.num_operations):
                                if np.isnan(std_group_limit[srv_id]):
                                    std_group_limit[srv_id] = max_limit
                            pprint({i: v for i, v in enumerate(std_group_limit)})

                    else:
                        std_group_limit = None

                F(test_stream, 'test', n_z, thresholds.get('val_threshold'), std_limit=std_group_limit)


if __name__ == '__main__':
    main()
