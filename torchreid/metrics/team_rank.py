from __future__ import absolute_import, division, print_function

import numpy as np


def evaluate_team_rank(
    distmat,
    q_pids,
    g_pids,
    q_action_ids,
    g_action_ids,
    q_teamids,
    g_teamids,
    max_rank=5,
    teammate_only=False,
):
    """Evaluate same-action, same-team retrieval.

    Args:
        teammate_only (bool): remove gallery images of the query identity.

    Returns:
        tuple: CMC curve, mean average precision, valid query count.
    """
    distmat = np.asarray(distmat)
    q_pids = np.asarray(q_pids)
    g_pids = np.asarray(g_pids)
    q_action_ids = np.asarray(q_action_ids)
    g_action_ids = np.asarray(g_action_ids)
    q_teamids = np.asarray(q_teamids)
    g_teamids = np.asarray(g_teamids)

    if distmat.shape != (q_pids.size, g_pids.size):
        raise ValueError(
            'distmat shape {} does not match {} queries and {} gallery '
            'samples'.format(distmat.shape, q_pids.size, g_pids.size)
        )
    if max_rank < 1:
        raise ValueError('max_rank must be positive')

    all_cmc = []
    all_ap = []

    for q_idx in range(q_pids.size):
        query_team = q_teamids[q_idx]
        if query_team < 0:
            continue

        candidate_mask = (
            (g_action_ids == q_action_ids[q_idx])
            & (g_teamids >= 0)
        )
        if teammate_only:
            candidate_mask &= g_pids != q_pids[q_idx]

        candidate_indices = np.flatnonzero(candidate_mask)
        if candidate_indices.size == 0:
            continue

        relevant = g_teamids[candidate_indices] == query_team
        if not np.any(relevant):
            continue

        local_order = np.argsort(
            distmat[q_idx, candidate_indices]
        )
        ranked_relevant = relevant[local_order].astype(np.int32)

        cmc = ranked_relevant.cumsum()
        cmc[cmc > 1] = 1
        padded_cmc = np.empty(max_rank, dtype=np.float32)
        visible_ranks = min(max_rank, cmc.size)
        padded_cmc[:visible_ranks] = cmc[:visible_ranks]
        if visible_ranks < max_rank:
            padded_cmc[visible_ranks:] = cmc[-1]
        all_cmc.append(padded_cmc)

        precision = ranked_relevant.cumsum() / (
            np.arange(ranked_relevant.size) + 1.0
        )
        average_precision = (
            precision * ranked_relevant
        ).sum() / ranked_relevant.sum()
        all_ap.append(average_precision)

    num_valid_queries = len(all_cmc)
    if num_valid_queries == 0:
        return (
            np.zeros(max_rank, dtype=np.float32),
            0.0,
            0,
        )

    return (
        np.mean(np.asarray(all_cmc), axis=0),
        float(np.mean(all_ap)),
        num_valid_queries,
    )
