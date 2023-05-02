import math
import os
import sys
import traceback
from functools import wraps
from typing import *

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score

from .fscore_utils import *

__all__ = ['analyze_anomaly_nll']


def analyze_anomaly_nll(nll_list: np.ndarray,
                        label_list: np.ndarray,
                        up_sample_normal: int = 1,
                        threshold: Optional[float] = None,
                        proba_cdf_file: Optional[str] = None,
                        auc_curve_file: Optional[str] = None,
                        method: Optional[str] = None,
                        dataset: Optional[str] = None,
                        save_dict: bool = False,
                        save_filename: str = 'baseline.csv'
                        ) -> Dict[str, float]:

    def log_error(method, default_value=None):
        @wraps(method)
        def wrapper(*args, **kwargs):
            try:
                return method(*args, **kwargs)
            except Exception:
                print(''.join(traceback.format_exception(*sys.exc_info())), file=sys.stderr)
                return default_value
        return wrapper

    def call_plot(fn_, *args, output_file, **kwargs):
        if output_file == ':show:':
            fig = fn_(*args, **kwargs)
            plt.show()
            plt.close()
        else:
            fn_(*args, output_file=output_file, **kwargs)

    # up sample normal nll & label if required
    if up_sample_normal and up_sample_normal > 1:
        normal_nll = nll_list[label_list == 0]
        normal_label = label_list[label_list == 0]
        nll_list = np.concatenate(
            [normal_nll] * (up_sample_normal - 1) + [nll_list],
            axis=0
        )
        label_list = np.concatenate(
            [normal_label] * (up_sample_normal - 1) + [label_list],
            axis=0
        )

    # prepare for analyze
    result_dict = {}
    is_anomaly_list = label_list != 0

    # separated nlls for different labels
    result_dict['nll_normal'] = float(np.mean(nll_list[label_list == 0]))
    result_dict['nll_drop'] = float(np.mean(nll_list[label_list == 1]))
    result_dict['nll_latency'] = float(np.mean(nll_list[label_list == 2]))

    # auc score
    result_dict['auc'] = float(auc_score(nll_list, is_anomaly_list))

    # best f-score
    F = log_error(best_fscore, default_value=(math.nan, math.nan))

    def best_fscore_for_label(label):
        not_label = 2 if label == 1 else 1
        mask = label_list != not_label
        return F(nll_list[mask], label_list[mask] != 0)

    best_fscore_total, _, best_pr_total, best_rc_total = F(nll_list, is_anomaly_list)
    best_fscore_drop, _, best_pr_drop, best_rc_drop = best_fscore_for_label(1)
    best_fscore_latency, best_threshold_latency, best_pr_latency, best_rc_latency = best_fscore_for_label(2)
    result_dict.update({
        'best_fscore': float(best_fscore_total),
        'best_fscore_drop': float(best_fscore_drop),
        'best_fscore_latency': float(best_fscore_latency),
        'best_pr': float(best_pr_total),
        'best_rc': float(best_rc_total),
        'best_pr_drop': float(best_pr_drop),
        'best_rc_drop': float(best_rc_drop),
        'best_pr_latency': float(best_pr_latency),
        'best_rc_latency': float(best_rc_latency),
        'best_threshold_latency': float(best_threshold_latency)
    })

    # f-score
    F = log_error(f1_score, default_value=math.nan)

    def fscore_for_label(label):
        not_label = 2 if label == 1 else 1
        mask = label_list != not_label
        return F(label_list[mask] != 0, nll_list[mask] > threshold)

    if threshold is not None:
        result_dict.update({
            'fscore': float(F(is_anomaly_list, nll_list > threshold)),
            'fscore_drop': float(fscore_for_label(1)),
            'fscore_latency': float(fscore_for_label(2)),
        })

    # save result
    if save_dict and method and dataset:
        dataset = dataset.rstrip('/')

        result_to_save = result_dict.copy()
        result_to_save['dataset'] = dataset
        result_to_save['method'] = method

        if os.path.exists(f'paper-data/{save_filename}'):
            df = pd.read_csv(f'paper-data/{save_filename}')
            
            if not df[(df['dataset']==dataset)&(df['method']==method)].empty:
                df.iloc[df[(df['dataset']==dataset)&(df['method']==method)].index[0]] = result_to_save
            else:
                df = df.append(result_to_save, ignore_index=True)
        else:
            df = pd.DataFrame()
            df = df.append(result_to_save, ignore_index=True)

        os.makedirs('paper-data', exist_ok=True)
        df.to_csv(f'paper-data/{save_filename}', index=False)

    return result_dict
