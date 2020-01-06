"""
Helper functions for replicating results
"""

import numpy as np
import torch
import tqdm
import multiprocessing as mp

from .utils import load_npz, largest_connected_components, preprocess_graph, train_val_test_split_tabular
from .GCN import GCN, GCN_Model
from .nettack import Nettack


def sparse_numpy2sparse_torch(x):
    x = x.tocoo()
    values = x.data
    indices = np.vstack((x.row, x.col)).astype(float)
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = x.shape
    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

class Evaluater:
    
    def __init__(self):
        pass
    
    def load_dataset(self, path):
        _A_obs, _X_obs, _z_obs = load_npz(path)
        # Normalizing Adjacency matrix
        _A_obs = _A_obs + _A_obs.T
        _A_obs[_A_obs > 1] = 1
        # For the algorithm to work, we have to consider a connected graph.
        lcc = largest_connected_components(_A_obs)
        _A_obs = _A_obs[lcc][:,lcc]
        _X_obs = _X_obs[lcc].astype('float32')
        _z_obs = _z_obs[lcc]
                
        assert np.abs(_A_obs - _A_obs.T).sum() == 0, "Input graph is not symmetric"
        assert _A_obs.max() == 1 and len(np.unique(_A_obs[_A_obs.nonzero()].A1)) == 1, "Graph must be unweighted"
        assert _A_obs.sum(0).A1.min() > 0, "Graph contains singleton nodes"
        
        self._A_obs = _A_obs
        self._X_obs = _X_obs
        self.A = sparse_numpy2sparse_torch(_A_obs)
        self.X = sparse_numpy2sparse_torch(_X_obs)
        self.N = _A_obs.shape[0]
        self.K = _z_obs.max()+1
        self.Z = _z_obs
        self.Ztorch = torch.tensor(_z_obs.astype(np.int64))
        # Normalizing adjacency matrix
        self.An = sparse_numpy2sparse_torch(preprocess_graph(_A_obs))
        self.degrees = _A_obs.sum(0).A1
        
    def create_splits(self):
        
        unlabeled_share = 0.8
        val_share = 0.1
        train_share = 1 - unlabeled_share - val_share
        splits = train_val_test_split_tabular(np.arange(self.N), train_size=train_share,
                                              val_size=val_share, test_size=unlabeled_share,
                                              stratify=self.Z)
        split_train, split_val, split_unlabeled = splits
        self.split_train = np.array(split_train).astype(np.int64)
        self.split_val = np.array(split_val).astype(np.int64)
        self.split_unlabeled = np.array(split_unlabeled).astype(np.int64)

        print(f'Number of training node : {len(split_train)}')
        print(f'Number of validation nodes : {len(split_val)}')
        print(f'Number of unlabeled (unknown) nodes : {len(split_unlabeled)}')
        
    def train_model(self, surrogate=False, with_perturb=False, disp=True, debug=False):
        sizes = [16, self.K]
        name = "surrogate" if surrogate else ("pertubed" if with_perturb else "clean")
        A = sparse_numpy2sparse_torch(self.nettack.adj_preprocessed) if (with_perturb and not surrogate) else self.An
        X = sparse_numpy2sparse_torch(self.nettack.X_obs) if (with_perturb and not surrogate) else self.X
        self.nn = GCN(sizes, A, X, with_relu=(not surrogate), name=name)
        self.model = GCN_Model(self.nn, lr=1e-2)
        self.model.train(self.split_train, self.split_val, self.Ztorch, print_info=disp, debug=debug)
        
        # Computing logits for every node
        self.model._compute_loss_and_backprop(np.arange(self.N), self.Ztorch, backward=False)
        self.logits = self.model.logit_nodes.detach().cpu().numpy()
        self.preds = self.logits.argmax(axis=1)
        truth = self.Z[self.split_val]
        if disp:
            print(f'Validation accuracy : {(self.Z[self.split_val] == self.preds[self.split_val]).mean():.2%}')
            print(f'Train accuracy : {(self.Z[self.split_train] == self.preds[self.split_train]).mean():.2%}')
            print(f'Unlabeled accuracy : {(self.Z[self.split_unlabeled] == self.preds[self.split_unlabeled]).mean():.2%}')
        if surrogate:
            self.W1 = self.model.gcn.gc1.weight
            self.W2 = self.model.gcn.gc2.weight
        self.probas = np.exp(self.logits) / np.exp(self.logits).sum(1)[:, None]
        probas_surr_sorted = np.argsort(-self.probas, axis=1)
        second_l = probas_surr_sorted[np.arange(self.N),
                                      (probas_surr_sorted == self.Z[:, None]).argmin(axis=1)]
        self.margins = (self.probas[np.arange(self.N), self.Z] -
                        self.probas[np.arange(self.N), second_l])
        
    def attack(self, u, verbose=False, n_perturbations=None, direct_attack=True,
               perturb_features=True, perturb_structure=True):
        self.nettack = Nettack(self._A_obs, self._X_obs, self.Z, self.W1, self.W2, u, verbose=verbose)
        self.nettack.reset()
        if n_perturbations is None:
            n_perturbations = int(self.degrees[u])
        n_influencers = 1 if direct_attack else 5
        self.nettack.attack_surrogate(n_perturbations,
                         perturb_structure=perturb_structure,
                         perturb_features=perturb_features,
                         direct=direct_attack,
                         n_influencers=n_influencers)

    def _produce_margin(self, data):
        n, id_, direct, features, structure = data
        self.attack(u=id_, verbose=False, direct_attack=direct, n_perturbations=int(self.degrees[id_] + 2),
              perturb_features=features, perturb_structure=structure)
        self.train_model(surrogate=False, with_perturb=True, disp=False)
        margin = self.margins[id_]
        return (n, id_, margin)

    def produce_margins(self, ids, direct=True, n_repeats=10, features=True,
                    structure=True, nb_process=5):
        dict_ids_i = dict(zip(ids, list(range(len(ids)))))
        data = [(n, id_, direct, features, structure)
                for n in range(n_repeats) for id_ in ids]
        if nb_process > 1:
            # raise ValueError('Multiprocessing does not work')
            with mp.Pool(processes=nb_process) as pool:
                raw_results = tqdm.tqdm_notebook(pool.map(self._produce_margin, data), total=len(ids)*n_repeats)
        else:
            raw_results = list(tqdm.tqdm_notebook(map(self._produce_margin, data), total=len(ids)*n_repeats))
        data = np.zeros((n_repeats, len(ids)))
        for x in raw_results:
            n, id_, m = x
            data[n, dict_ids_i[id_]] = m
        return data

# def produce_margin(data):
#     Ev, n, id_, direct, features, structure = data
#     Ev.attack(u=id_, verbose=False, direct_attack=direct, n_perturbations=int(Ev.degrees[id_] + 2),
#               perturb_features=features, perturb_structure=structure)
#     Ev.train_model(surrogate=False, with_perturb=True, disp=False)
#     margin = Ev.margins[id_]
#     return (n, id_, margin)

# def produce_margins(Ev, ids, direct=True, n_repeats=10, features=True,
#                     structure=True, nb_process=5):
#     pbar = tqdm.tqdm_notebook(total=n_repeats * len(ids))

#     data = [(Ev, n, id_, direct, features, structure)
#             for n in range(n_repeats) for id_ in ids]
#     with mp.Pool(processes=nb_process) as pool:
#         raw_results = tqdm.tqdm_notebook(pool.map(produce_margin, data), total=len(ids)*n_repeats)
#     return raw_results
    # margins = []
    # for i in range(n_repeats):
    #     margins_i = []
    #     for id_ in ids:
    #         Ev.attack(u=i, verbose=False, direct_attack=direct, n_perturbations=int(Ev.degrees[i] + 2),
    #                   perturb_features=features, perturb_structure=structure)
    #         Ev.train_model(surrogate=False, with_perturb=True, disp=False)
    #         pbar.update(1)
    #         margins_i.append(Ev.margins[id_])
    #     margins.append(margins_i)
    # pbar.close()
    # return np.array(margins)


