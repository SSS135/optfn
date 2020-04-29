"""
Random search - presented as hyperopt.fmin_random
"""
from __future__ import absolute_import
import logging
from collections import OrderedDict

import numpy as np
from hyperopt import rand

from hyperopt import pyll

from hyperopt.base import miscs_update_idxs_vals
from sklearn.gaussian_process import GaussianProcessRegressor

logger = logging.getLogger(__name__)


def suggest(new_ids, domain, trials, seed, samples_count=200, maxlog=8, rand_iters=3,
            create_regressor=lambda: GaussianProcessRegressor(alpha=1e-6)):
    rand_iters = max(1, rand_iters)
    if len(trials) < rand_iters:
        return rand.suggest(new_ids, domain, trials, seed)

    rng = np.random.RandomState(seed)
    rval = []
    for ii, new_id in enumerate(new_ids):
        # -- sample new specs, idxs, vals
        idxs, vals = get_best_eval(new_id, domain, trials, rng, samples_count, maxlog, create_regressor)
        new_result = domain.new_result()
        new_misc = dict(tid=new_id, cmd=domain.cmd, workdir=domain.workdir)
        miscs_update_idxs_vals([new_misc], idxs, vals)
        rval.extend(trials.new_trial_docs([new_id],
                    [None], [new_result], [new_misc]))
    return rval


# def suggest_batch(new_ids, domain, trials, seed):
#     rng = np.random.RandomState(seed)
#     # -- sample new specs, idxs, vals
#     idxs, vals = pyll.rec_eval(
#         domain.s_idxs_vals,
#         memo={
#             domain.s_new_ids: new_ids,
#             domain.s_rng: rng,
#         })
#     return idxs, vals


def get_best_eval(new_id, domain, trials, rng, samples_count, maxlog, create_regressor):
    idxs_vals = [get_random_idxs_vals(new_id, domain, rng) for _ in range(samples_count)]
    x_train, y_train, means = get_train_data(trials)
    x_test = get_test_data(idxs_vals, trials, means)

    reg = create_regressor()
    reg.fit(x_train, y_train)
    y_pred = reg.predict(x_test)

    choice_weights = np.logspace(maxlog, 0, num=samples_count)
    choice_weights /= np.sum(choice_weights)

    min_idx = rng.choice(np.argsort(y_pred), p=choice_weights)

    return idxs_vals[min_idx]


def get_random_idxs_vals(new_id, domain, rng):
    return pyll.rec_eval(
        domain.s_idxs_vals,
        memo={
            domain.s_new_ids: [new_id],
            domain.s_rng: rng,
        })


def get_test_data(idxs_vals, trials, means):
    param_names = list(trials.idxs.keys())

    idxs = dict()
    vals = dict()
    for name in param_names:
        idxs[name], vals[name] = ([], [])
    for sample_index in range(len(idxs_vals)):
        _, val = idxs_vals[sample_index]
        for n in param_names:
            if len(val[n]) == 1:
                idxs[n].append(sample_index)
                vals[n].append(val[n][0])

    vals = list(OrderedDict(sorted(vals.items(), key=lambda t: t[0])).values())
    idxs = list(OrderedDict(sorted(idxs.items(), key=lambda t: t[0])).values())

    return get_x_data(idxs, vals, means)[0]


def get_train_data(trials):
    train_idxvals = get_idxs_vals(trials)
    x, means = get_x_data(*train_idxvals)
    y = get_y_data(trials)
    return x, y, means


def get_idxs_vals(trials):
    vals = list(OrderedDict(sorted(trials.vals.items(), key=lambda t: t[0])).values())
    idxs = list(OrderedDict(sorted(trials.idxs.items(), key=lambda t: t[0])).values())
    return idxs, vals


def get_x_data(idxs, vals, means=None):
    samples_count = int(np.max(np.concatenate(idxs)) + 1)
    param_count = len(idxs)
    if means is None:
        means = np.nan_to_num([np.mean(v) if len(v) > 0 else 0 for v in vals])
    samples = np.zeros((param_count, samples_count))
    for i in range(param_count):
        samples[i] = means[i]
        samples[i, idxs[i]] = vals[i]
    return samples.T, means


def get_y_data(trials):
    return np.array([res['loss'] for res in trials.results])


# flake8 likes no trailing blank line