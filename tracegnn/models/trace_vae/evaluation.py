import json
import math
from pprint import pprint
from typing import *

import mltk
import tensorkit as tk
import yaml
from tensorkit import tensor as T
from tqdm import tqdm
import pickle
import snappy
import numpy as np
import os

from tracegnn.utils import *
from tracegnn.data import *
from ...data import TraceGraph, TraceGraphNode
from ...utils import TraceGraphLatencyRangeFile
from .graph_utils import p_net_to_trace_graphs, trace_graph_key
from .model import TraceVAE
from .tensor_utils import *
from .types import TraceGraphBatch

__all__ = [
    'do_evaluate_nll',
    'do_evaluate_prior',
    'do_anomaly_detect'
]


def do_evaluate_nll(test_stream: mltk.DataStream,
                    vae: TraceVAE,
                    id_manager: TraceGraphIDManager,
                    latency_range: TraceGraphLatencyRangeFile,
                    n_z: int,
                    use_biased: bool = True,
                    use_latency_biased: bool = True,
                    no_latency: bool = False,
                    no_struct: bool = False,
                    std_limit: Optional[T.Tensor] = None,
                    latency_log_prob_weight: bool = False,
                    latency_logstd_min: Optional[float] = None,
                    test_threshold: Optional[float] = None,
                    test_loop=None,
                    summary_writer=None,
                    clip_nll=None,
                    use_embeddings: bool = False,
                    num_embedding_samples=None,
                    nll_output_file=None,
                    proba_cdf_file=None,
                    auc_curve_file=None,
                    latency_hist_file=None,
                    operation_id_dict_out=None,  # corresponding to latency_std_dict_out
                    latency_std_dict_out=None,
                    latency_reldiff_dict_out=None,
                    p_node_count_dict_out=None,
                    p_edge_dict_out=None,
                    latency_dict_prefix='',
                    ):
    # check params
    if std_limit is not None:
        std_limit = T.as_tensor(std_limit, dtype=T.float32)

    # result buffer
    nll_list = []
    label_list = []
    trace_id_list = []
    graph_key_list = []
    z_buffer = []  # the z embedding buffer of the graph
    z2_buffer = []  # the z2 embedding buffer of the graph
    z_label = []  # the label for z and z2
    latency_samples = {}
    result_dict = {}

    if operation_id_dict_out is not None:
        for key in ('normal', 'drop', 'latency'):
            if key not in operation_id_dict_out:
                operation_id_dict_out[latency_dict_prefix + key] = ArrayBuffer(81920, dtype=np.int)

    if latency_std_dict_out is not None:
        for key in ('normal', 'drop', 'latency'):
            if key not in latency_std_dict_out:
                latency_std_dict_out[latency_dict_prefix + key] = ArrayBuffer(81920)

    if latency_reldiff_dict_out is not None:
        for key in ('normal', 'drop', 'latency'):
            if key not in latency_reldiff_dict_out:
                latency_reldiff_dict_out[latency_dict_prefix + key] = ArrayBuffer(81920)

    if p_node_count_dict_out is not None:
        for key in ('normal', 'drop', 'latency'):
            if key not in p_node_count_dict_out:
                p_node_count_dict_out[latency_dict_prefix + key] = ArrayBuffer(81920)

    if p_edge_dict_out is not None:
        for key in ('normal', 'drop', 'latency'):
            if key not in p_edge_dict_out:
                p_edge_dict_out[latency_dict_prefix + key] = ArrayBuffer(81920)

    def add_embedding(buffer, label, tag, limit=None):
        if limit is not None:
            indices = np.arange(len(buffer))
            np.random.shuffle(indices)
            indices = indices[:limit]
            buffer = buffer[indices]
            label = label[indices]
        summary_writer.add_embedding(
            buffer,
            metadata=label,
            tag=tag,
        )

    # run evaluation
    def eval_step(trace_graphs: List[TraceGraph]):
        G = TraceGraphBatch(
            id_manager=id_manager,
            latency_range=latency_range,
            trace_graphs=trace_graphs,
        )
        chain = vae.q(G, n_z=n_z, no_latency=no_latency).chain(
            vae.p,
            latent_axis=0,
            G=G,
            use_biased=use_biased,
            use_latency_biased=use_latency_biased,
            no_latency=no_latency,
            latency_logstd_min=latency_logstd_min,
            latency_log_prob_weight=latency_log_prob_weight,
            std_limit=std_limit,
        )
        if no_struct:
            q, p = chain.q, chain.p
            del q['z']
            del p['z']
            del p['adj']
            del p['node_count']
            del p['node_type']
            chain = q.chain(lambda *args, **kwargs: p, latent_axis=0)

        loss = chain.vi.training.sgvb()
        nll = -chain.vi.evaluation.is_loglikelihood()

        # clip the nll, and treat 'NaN' or 'Inf' nlls as `config.test.clip_nll`
        if clip_nll is not None:
            clip_limit = T.float_scalar(clip_nll)
            loss = T.where(loss < clip_limit, loss, clip_limit)
            nll = T.where(nll < clip_limit, nll, clip_limit)

        # the nlls and labels of this step
        step_label_list = np.array([
            0 if not g.data.get('is_anomaly') else (
                1 if g.data['anomaly_type'] == 'drop' else 2)
            for g in trace_graphs
        ])

        # Load the graph_key
        step_graph_key_list = [trace_graph_key(g) for g in trace_graphs]
        step_trace_id_list = [g.trace_id for g in trace_graphs]

        if not no_struct:
            # collect operation id
            if operation_id_dict_out is not None:
                collect_operation_id(operation_id_dict_out[f'{latency_dict_prefix}normal'], chain, step_label_list == 0)
                collect_operation_id(operation_id_dict_out[f'{latency_dict_prefix}drop'], chain, step_label_list == 1)
                collect_operation_id(operation_id_dict_out[f'{latency_dict_prefix}latency'], chain, step_label_list == 2)

            # collect latency
            if latency_std_dict_out is not None:
                collect_latency_std(latency_std_dict_out[f'{latency_dict_prefix}normal'], chain, step_label_list == 0)
                collect_latency_std(latency_std_dict_out[f'{latency_dict_prefix}drop'], chain, step_label_list == 1)
                collect_latency_std(latency_std_dict_out[f'{latency_dict_prefix}latency'], chain, step_label_list == 2)

            # collect relative diff
            if latency_reldiff_dict_out is not None:
                collect_latency_reldiff(latency_reldiff_dict_out[f'{latency_dict_prefix}normal'], chain, step_label_list == 0)
                collect_latency_reldiff(latency_reldiff_dict_out[f'{latency_dict_prefix}drop'], chain, step_label_list == 1)
                collect_latency_reldiff(latency_reldiff_dict_out[f'{latency_dict_prefix}latency'], chain, step_label_list == 2)

            # collect p node count
            if p_node_count_dict_out is not None:
                collect_p_node_count(p_node_count_dict_out[f'{latency_dict_prefix}normal'], chain, step_label_list == 0)
                collect_p_node_count(p_node_count_dict_out[f'{latency_dict_prefix}drop'], chain, step_label_list == 1)
                collect_p_node_count(p_node_count_dict_out[f'{latency_dict_prefix}latency'], chain, step_label_list == 2)

            # collect p edge
            if p_edge_dict_out is not None:
                collect_p_edge(p_edge_dict_out[f'{latency_dict_prefix}normal'], chain, step_label_list == 0)
                collect_p_edge(p_edge_dict_out[f'{latency_dict_prefix}drop'], chain, step_label_list == 1)
                collect_p_edge(p_edge_dict_out[f'{latency_dict_prefix}latency'], chain, step_label_list == 2)

            # inspect the internals of every trace graph
            if 'latency' in chain.p:
                p_latency = chain.p['latency'].distribution.base_distribution
                p_latency_mu, p_latency_std = p_latency.mean, p_latency.std
                if len(T.shape(p_latency.mean)) == 4:
                    p_latency_mu = p_latency_mu[0]
                    p_latency_std = p_latency_std[0]

                latency_sample = T.to_numpy(T.random.normal(p_latency_mu, p_latency_std))

                for i, tg in enumerate(trace_graphs):
                    assert isinstance(tg, TraceGraph)
                    if step_label_list[i] == 0:
                        for j in range(tg.node_count):
                            node_type = int(T.to_numpy(G.dgl_graphs[i].ndata['node_type'][j]))
                            if node_type not in latency_samples:
                                latency_samples[node_type] = []
                            mu, std = latency_range[node_type]
                            latency_samples[node_type].append(latency_sample[i, j, 0] * std + mu)

            if use_embeddings:
                for i in range(len(trace_graphs)):
                    if step_label_list[i] == 0:
                        node_type = trace_graphs[i].root.operation_id
                        node_label = id_manager.operation_id.reverse_map(node_type)
                        z_label.append(node_label)
                        z_buffer.append(T.to_numpy(chain.q['z'].tensor[0, i]))
                        if 'z2' in chain.q:
                            z2_buffer.append(T.to_numpy(chain.q['z2'].tensor[0, i]))

        # memorize the outputs
        nll_list.extend(T.to_numpy(nll))
        label_list.extend(step_label_list)
        trace_id_list.extend(step_trace_id_list)
        graph_key_list.extend(step_graph_key_list)

        # return a dict of the test result
        ret = {}
        normal_losses = T.to_numpy(loss)[step_label_list == 0]
        if len(normal_losses) > 0:
            test_loss = np.nanmean(normal_losses)
            if not math.isnan(test_loss):
                ret['loss'] = test_loss
        return ret

    with T.no_grad():
        # run test on test set
        if test_loop is not None:
            with test_loop.timeit('eval_time'):
                r = test_loop.run(eval_step, test_stream)
                if 'loss' in r:
                    r['test_loss'] = r['loss']
                if 'test_loss' in r:
                    result_dict['test_loss'] = r['test_loss']
        else:
            test_losses = []
            test_weights = []
            for [trace_graphs] in tqdm(test_stream, total=test_stream.batch_count):
                r = eval_step(trace_graphs)
                if 'loss' in r:
                    test_losses.append(r['loss'])
                    test_weights.append(len(trace_graphs))
            test_weights = np.asarray(test_weights)
            result_dict['test_loss'] = np.sum(
                np.asarray(test_losses) *
                (test_weights / np.sum(test_weights))
            )

        # save the evaluation results
        nll_list = np.asarray(nll_list)
        label_list = np.asarray(label_list)
        graph_key_list = np.asarray(pickle.dumps(graph_key_list))

        # analyze nll
        result_dict.update(
            analyze_anomaly_nll(
                nll_list=nll_list,
                label_list=label_list,
                proba_cdf_file=proba_cdf_file,
                auc_curve_file=auc_curve_file,
                threshold=test_threshold,
            )
        )

        if nll_output_file is not None:
            np.savez_compressed(
                ensure_parent_exists(nll_output_file),
                nll_list=nll_list,
                label_list=label_list,
                graph_key_list=graph_key_list,
                anomaly_degree=nll_list / result_dict['best_threshold_latency']
            )

            print(f'{latency_dict_prefix} file saved to {nll_output_file}')

        # z embedding
        if use_embeddings:
            # add the operation embedding
            operation_buffer = T.to_numpy(vae.operation_embedding(
                T.arange(0, id_manager.num_operations, dtype=T.int64)))
            operation_label = [
                id_manager.operation_id.reverse_map(i)
                for i in range(id_manager.num_operations)
            ]
            add_embedding(operation_buffer, operation_label, 'operation')

            # add z & z2 embedding
            z_label = np.stack(z_label, axis=0)
            add_embedding(
                np.stack(z_buffer, axis=0),
                z_label,
                tag='z',
                limit=num_embedding_samples
            )
            if z2_buffer:
                add_embedding(
                    np.stack(z2_buffer, axis=0),
                    z_label,
                    tag='z2',
                    limit=num_embedding_samples
                )

    # return the results
    result_dict = {k: float(v) for k, v in result_dict.items()}
    return result_dict


def do_evaluate_prior(vae: TraceVAE,
                      id_manager: TraceGraphIDManager,
                      latency_range: TraceGraphLatencyRangeFile,
                      n_samples: int,
                      batch_size: int,
                      eval_n_z: int,
                      nll_threshold: Optional[float] = None,
                      use_biased: bool = True,
                      output_file: Optional[str] = None,
                      latency_hist_out: Optional[str] = None,
                      ):
    with T.no_grad():
        # results
        sample_count = 0
        drop_count = 0
        result_dict = {}
        latency_map = {}

        def add_sample(g: TraceGraph):
            if latency_hist_out is not None:
                for _, nd in g.iter_bfs():
                    assert isinstance(nd, TraceGraphNode)
                    if nd.operation_id not in latency_map:
                        latency_map[nd.operation_id] = []
                    latency_map[nd.operation_id].append(nd.features.avg_latency)

        # run by sample from prior
        n_batches = (n_samples + batch_size - 1) // batch_size
        for _ in tqdm(range(n_batches), total=n_batches, desc='Sample graphs from prior'):
            # sample from prior
            p = vae.p(n_z=batch_size)
            trace_graphs = p_net_to_trace_graphs(
                p,
                id_manager=id_manager,
                latency_range=latency_range,
                discard_node_with_type_0=True,
                discard_node_with_unknown_latency_range=True,
                discard_graph_with_error_node_count=True,
            )

            sample_count += len(trace_graphs)
            drop_count += sum(g is None for g in trace_graphs)
            trace_graphs = [g for g in trace_graphs if g is not None]

            # evaluate the NLLs
            G = TraceGraphBatch(
                id_manager=id_manager,
                latency_range=latency_range,
                trace_graphs=trace_graphs,
            )
            chain = vae.q(G=G, n_z=eval_n_z). \
                chain(vae.p, n_z=eval_n_z, latent_axis=0, use_biased=use_biased)
            eval_nlls = T.to_numpy(chain.vi.evaluation.is_loglikelihood(reduction='none'))

            # purge too-low NLL graphs
            for g, nll in zip(trace_graphs, eval_nlls):
                if nll >= nll_threshold:
                    drop_count += 1
                else:
                    add_sample(g)

    # save the results
    drop_rate = float(drop_count / sample_count)
    result_dict.update({
        'drop_rate': drop_rate,
    })
    pprint(result_dict)

    if output_file is not None:
        _, ext = os.path.splitext(output_file)
        if ext == '.json':
            result_cont = json.dumps(result_dict)
        else:
            result_cont = yaml.safe_dump(result_dict)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result_cont)


def do_anomaly_detect(test_stream: mltk.DataStream,
                    vae: TraceVAE,
                    id_manager: TraceGraphIDManager,
                    latency_range: TraceGraphLatencyRangeFile,
                    n_z: int,
                    use_biased: bool = True,
                    use_latency_biased: bool = True,
                    no_latency: bool = False,
                    no_struct: bool = False,
                    std_limit: Optional[T.Tensor] = None,
                    latency_log_prob_weight: bool = False,
                    latency_logstd_min: Optional[float] = None,
                    test_threshold: Optional[float] = None,
                    test_loop=None,
                    summary_writer=None,
                    clip_nll=None,
                    use_embeddings: bool = False,
                    num_embedding_samples=None,
                    nll_output_file=None,
                    proba_cdf_file=None,
                    auc_curve_file=None,
                    latency_hist_file=None,
                    operation_id_dict_out=None,  # corresponding to latency_std_dict_out
                    latency_std_dict_out=None,
                    latency_reldiff_dict_out=None,
                    p_node_count_dict_out=None,
                    p_edge_dict_out=None,
                    latency_dict_prefix='',
                    ):
    # check params
    if std_limit is not None:
        std_limit = T.as_tensor(std_limit, dtype=T.float32)

    # result buffer
    nll_list = []
    label_list = []
    graph_key_list = []
    z_buffer = []  # the z embedding buffer of the graph
    z2_buffer = []  # the z2 embedding buffer of the graph
    z_label = []  # the label for z and z2

    def add_embedding(buffer, label, tag, limit=None):
        if limit is not None:
            indices = np.arange(len(buffer))
            np.random.shuffle(indices)
            indices = indices[:limit]
            buffer = buffer[indices]
            label = label[indices]
        summary_writer.add_embedding(
            buffer,
            metadata=label,
            tag=tag,
        )

    # run evaluation
    def eval_step(trace_graphs):
        G = TraceGraphBatch(
            id_manager=id_manager,
            latency_range=latency_range,
            trace_graphs=trace_graphs,
        )
        chain = vae.q(G, n_z=n_z, no_latency=no_latency).chain(
            vae.p,
            latent_axis=0,
            G=G,
            use_biased=use_biased,
            use_latency_biased=use_latency_biased,
            no_latency=no_latency,
            latency_logstd_min=latency_logstd_min,
            latency_log_prob_weight=latency_log_prob_weight,
            std_limit=std_limit,
        )
        if no_struct:
            q, p = chain.q, chain.p
            del q['z']
            del p['z']
            del p['adj']
            del p['node_count']
            del p['node_type']
            chain = q.chain(lambda *args, **kwargs: p, latent_axis=0)

        loss = chain.vi.training.sgvb()
        nll = -chain.vi.evaluation.is_loglikelihood()

        # clip the nll, and treat 'NaN' or 'Inf' nlls as `config.test.clip_nll`
        if clip_nll is not None:
            clip_limit = T.float_scalar(clip_nll)
            loss = T.where(loss < clip_limit, loss, clip_limit)
            nll = T.where(nll < clip_limit, nll, clip_limit)

        # the nlls and labels of this step
        step_label_list = np.array([
            0 if not g.data.get('is_anomaly') else (
                1 if g.data['anomaly_type'] == 'drop' else 2)
            for g in trace_graphs
        ])

        # Load the graph_key
        step_graph_key_list = [trace_graph_key(g) for g in trace_graphs]

        if not no_struct:
            if use_embeddings:
                for i in range(len(trace_graphs)):
                    if step_label_list[i] == 0:
                        node_type = trace_graphs[i].root.operation_id
                        node_label = id_manager.operation_id.reverse_map(node_type)
                        z_label.append(node_label)
                        z_buffer.append(T.to_numpy(chain.q['z'].tensor[0, i]))
                        if 'z2' in chain.q:
                            z2_buffer.append(T.to_numpy(chain.q['z2'].tensor[0, i]))

        # memorize the outputs
        nll_list.extend(T.to_numpy(nll))
        label_list.extend(step_label_list)
        graph_key_list.extend(step_graph_key_list)

        # return a dict of the test result
        ret = {}
        normal_losses = T.to_numpy(loss)[step_label_list == 0]
        if len(normal_losses) > 0:
            test_loss = np.nanmean(normal_losses)
            if not math.isnan(test_loss):
                ret['loss'] = test_loss
        return ret

    with T.no_grad():
        # run test on test set
        if test_loop is not None:
            with test_loop.timeit('eval_time'):
                r = test_loop.run(eval_step, test_stream)
                if 'loss' in r:
                    r['test_loss'] = r['loss']
        else:
            test_losses = []
            test_weights = []
            for [trace_graphs] in tqdm(test_stream, total=test_stream.batch_count):
                r = eval_step(trace_graphs)
                if 'loss' in r:
                    test_losses.append(r['loss'])
                    test_weights.append(len(trace_graphs))
            test_weights = np.asarray(test_weights)

        # save the evaluation results
        nll_list = np.asarray(nll_list)
        label_list = np.asarray(label_list)
        graph_key_list = np.asarray(pickle.dumps(graph_key_list))

        # z embedding
        if use_embeddings:
            # add the operation embedding
            operation_buffer = T.to_numpy(vae.operation_embedding(
                T.arange(0, id_manager.num_operations, dtype=T.int64)))
            operation_label = [
                id_manager.operation_id.reverse_map(i)
                for i in range(id_manager.num_operations)
            ]
            add_embedding(operation_buffer, operation_label, 'operation')

            # add z & z2 embedding
            z_label = np.stack(z_label, axis=0)
            add_embedding(
                np.stack(z_buffer, axis=0),
                z_label,
                tag='z',
                limit=num_embedding_samples
            )
            if z2_buffer:
                add_embedding(
                    np.stack(z2_buffer, axis=0),
                    z_label,
                    tag='z2',
                    limit=num_embedding_samples
                )
