"""
Purpose: Visualization Library
Author: Andy
"""


import numpy as np

from tqdm import tqdm
from scipy.stats import wasserstein_distance as emd


def calc_mse(preds, truths):

    output = np.mean((preds - truths) ** 2)

    return output


def calc_mae(preds, truths):

    output = np.mean(np.abs(preds - truths))

    return output


def calc_emd(preds, truths):

    preds = preds.reshape(-1)
    truths = truths.reshape(-1)

    return emd(preds, truths)


def compare_many(preds, truths, measure):

    results = [measure(p, t) for p, t in zip(preds, truths)]

    return np.mean(results)

def get_all_measures(all_versions, all_results):

    final_results = {}

    pbar = tqdm(total=len(all_versions * 2), desc="Measures")

    for tag in ["real", "imag"]:

        t_results = {}

        for version in all_versions:

            #v_results = {}

            if version == 0:
                name = "rnn"
            elif version == 1:
                name = "lstm"
            elif version == 2:
                name = "convlstm"
            else:
                raise NotImplementedError

            t_results[name] = {} 

            results = all_results["version_%s" % version]
            all_truths, all_preds = results["truths"], results["preds"]

            if tag == "real":
                preds = all_preds[:, :, 0, :, :]
                truths = all_truths[:, :, 0, :, :]
            elif tag == "imag":
                preds = all_preds[:, :, 1, :, :]
                truths = all_truths[:, :, 1, :, :]

            else:

                raise NotImplementedError
            t_results[name]["mse"] = compare_many(preds, truths, calc_mse)
            t_results[name]["mae"] = compare_many(preds, truths, calc_mae)
            t_results[name]["emd"] = compare_many(preds, truths, calc_emd)

            #t_results.append(v_results)
            pbar.update(1)
        
        final_results[tag] = t_results

    pbar.close()

    return final_results
