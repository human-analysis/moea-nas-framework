import random
import numpy as np

import torch

_DEBUG = False


def reset_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_correlation(prediction, target):
    import scipy.stats as stats

    rmse = np.sqrt(((prediction - target) ** 2).mean())

    try:
        r, _ = stats.pearsonr(prediction[:, 0], target[:, 0])
    except IndexError:
        r, _ = stats.pearsonr(prediction, target)

    rho, _ = stats.spearmanr(prediction, target)
    tau, _ = stats.kendalltau(prediction, target)

    return rmse, r, rho, tau


if __name__ == '__main__':
    import json
    from search.search_spaces.ofa_search_space import BasicSearchSpace
    from search.surrogate_models import SurrogateModel

    basic_data = json.load(open("../data/ofa_fanet_basic_rtx_fps@0.5.json", "r"))

    search_space = BasicSearchSpace()

    subnet_str = [d['config'] for d in basic_data]
    features = search_space.features(search_space.encode(subnet_str))
    latencies = np.array([d['latency'] for d in basic_data])
    print(len(latencies))

    perm = np.random.permutation(len(latencies))
    predictor = SurrogateModel('mlp')
    pool_of_experts = predictor.fit(features[perm[:1000]], latencies[perm[:1000]], ensemble=True)

    pred = predictor.predict(pool_of_experts, features[perm[1000:]])
    rmse, r, rho, tau = get_correlation(pred, latencies[perm[1000:]])
    print("rmse: {:.4f}, pearson: {:.4f}, spearman: {:.4f}, kendall: {:.4f}".format(rmse, r, rho, tau))

