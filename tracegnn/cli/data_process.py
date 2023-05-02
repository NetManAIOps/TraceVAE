import math
import pickle as pkl
import random
import shutil
import sys

import click
import numpy as np
from tqdm import tqdm

from tracegnn.constants import *
from tracegnn.data import *
from tracegnn.utils import *


def get_graph_key(g):
    node_types = set()
    stack = [g.root]
    while stack:
        nd = stack.pop()
        node_types.add(nd.operation_id)
        stack.extend(nd.children)
    return g.root.operation_id, g.max_depth, tuple(sorted(node_types))


@click.group()
def main():
    pass


@main.command()
@click.option('-i', '--input-dir')
@click.option('-o', '--output-dir')
@click.option('-n', '--name', type=str, required=True)
def csv_to_db(input_dir, output_dir, name):
    # check the parameters
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)

    input_path = os.path.join(input_dir, f"{name}.csv")
    output_path = os.path.join(output_dir, "processed", name)

    # Load id_manager
    id_manager = TraceGraphIDManager(os.path.join(input_dir, 'id_manager'))

    # process the traces
    # load the graphs
    if 'test' not in name:
        df = load_trace_csv(input_path)
        trace_graphs = df_to_trace_graphs(
            df,
            id_manager=id_manager,
            merge_spans=True,
        )

        # write to db
        if os.path.exists(output_path):
            shutil.rmtree(output_path)

        db = BytesSqliteDB(output_path, write=True)
        with db, db.write_batch():
            for g in tqdm(trace_graphs, desc='Save graphs'):
                db.add(g.to_bytes())
    else:
        # read test data
        df = load_trace_csv(input_path, is_test=True)

        for i in range(3):
            trace_graphs = df_to_trace_graphs(
                df,
                id_manager=id_manager,
                merge_spans=True,
                test_label=i
            )

            # write to db
            if i == 0:
                output_path = os.path.join(output_dir, 'processed', 'test')
            elif i == 1:
                output_path = os.path.join(output_dir, 'processed', 'test-drop')
            else:
                output_path = os.path.join(output_dir, 'processed', 'test-latency')

            if os.path.exists(output_path):
                shutil.rmtree(output_path)

            db = BytesSqliteDB(output_path, write=True)
            with db, db.write_batch():
                for g in tqdm(trace_graphs, desc='Save graphs'):
                    db.add(g.to_bytes())


@main.command()
@click.option('-i', '--input-dir')
@click.option('-o', '--output_dir')
def preprocess(input_dir, output_dir):
    print("Convert datasets...")
    print("------------> Train")
    os.system(f"python3 -m tracegnn.cli.data_process csv-to-db -i {input_dir} -o {output_dir} -n train")
    print("------------> Val")
    os.system(f"python3 -m tracegnn.cli.data_process csv-to-db -i {input_dir} -o {output_dir} -n val")
    print("------------> Test")
    os.system(f"python3 -m tracegnn.cli.data_process csv-to-db -i {input_dir} -o {output_dir} -n test")

    print("Finished!")

if __name__ == '__main__':
    main()
