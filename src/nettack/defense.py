"""
Functions for making a graph more robust
"""
import numpy as np
import scipy.sparse as sp
import tqdm

from .nettack import Nettack
from .GCN import GCN, GCN_Model
from .replication import Evaluater, sparse_numpy2sparse_torch

class Netdef(Nettack):
    
    
    def feature_scores(self):
        """
        Compute feature scores for all possible feature changes.
        """

        if self.cooc_constraint is None:
            self.compute_cooccurrence_constraint(self.influencer_nodes)
        logits = self.compute_logits()
        best_wrong_class = self.strongest_wrong_class(logits)
        gradient = self.gradient_wrt_x(self.label_u) - self.gradient_wrt_x(best_wrong_class)
        surrogate_loss = logits[self.label_u] - logits[best_wrong_class]

        gradients_flipped = (gradient * -1).tolil()
        gradients_flipped[self.X_obs.nonzero()] *= -1

        X_influencers = sp.lil_matrix(self.X_obs.shape)
        X_influencers[self.influencer_nodes] = self.X_obs[self.influencer_nodes]
        gradients_flipped = gradients_flipped.multiply((self.cooc_constraint + X_influencers) > 0)
        nnz_ixs = np.array(gradients_flipped.nonzero()).T

        sorting = np.argsort(gradients_flipped[tuple(nnz_ixs.T)]).A1
        sorted_ixs = nnz_ixs[sorting]
        grads = gradients_flipped[tuple(nnz_ixs[sorting].T)]

        scores = surrogate_loss - grads
        # return sorted_ixs[::-1], scores.A1[::-1]
        return sorted_ixs, scores.A1

    def struct_score(self, a_hat_uv, XW):
        """
        Compute structure scores, cf. Eq. 15 in the paper
        Parameters
        ----------
        a_hat_uv: sp.sparse_matrix, shape [P,2]
            Entries of matrix A_hat^2_u for each potential edge (see paper for explanation)
        XW: sp.sparse_matrix, shape [N, K], dtype float
            The class logits for each node.
        Returns
        -------
        np.array [P,]
            The struct score for every row in a_hat_uv
        """

        logits = a_hat_uv.dot(XW)
        label_onehot = np.eye(XW.shape[1])[self.label_u]
        best_wrong_class_logits = (logits - 1000 * label_onehot).max(1)
        logits_for_correct_class = logits[:,self.label_u]
        struct_scores = logits_for_correct_class - best_wrong_class_logits
        # return struct_scores
        return - struct_scores


# Just Nettack with Loss opposed
class EvaluaterDef(Evaluater):
    
    def defend(self, u, verbose=False, n_perturbations=None, direct_attack=True,
               perturb_features=True, perturb_structure=True):
        self.nettack = Netdef(self._A_obs, self._X_obs, self.Z, self.W1, self.W2, u, verbose=verbose)
        self.nettack.reset()
        if n_perturbations is None:
            n_perturbations = int(self.degrees[u])
        n_influencers = 1 if direct_attack else 5
        self.nettack.attack_surrogate(n_perturbations,
                         perturb_structure=perturb_structure,
                         perturb_features=perturb_features,
                         direct=direct_attack,
                         n_influencers=n_influencers)


def margin_attack(Ev, A, X):
    nn = GCN([16, Ev.K], A, X, with_relu=True)
    model = GCN_Model(nn, lr=1e-2)
    model.train(Ev.split_train, Ev.split_val, Ev.Ztorch, print_info=False, debug=False)
    model._compute_loss_and_backprop(np.arange(Ev.N), Ev.Ztorch, backward=False)
    logits = model.logit_nodes.detach().cpu().numpy()
    probas = np.exp(logits) / np.exp(logits).sum(1)[:, None]
    probas_surr_sorted = np.argsort(-probas, axis=1)
    second_l = probas_surr_sorted[np.arange(Ev.N), (probas_surr_sorted == Ev.Z[:, None]).argmin(axis=1)]
    margins = (probas[np.arange(Ev.N), Ev.Z] - probas[np.arange(Ev.N), second_l])
    return margins[Ev.nettack.u]


def evaluate_graph_optim(Ev, EvDef, n_nodes=10, n_retrain=5):
    nodes = np.random.choice(Ev.split_unlabeled, size=n_nodes, replace=False)
    pbar = tqdm.tqdm_notebook(total=n_nodes*n_retrain)
    margins = np.zeros((n_nodes, n_retrain, 4))
    for i, node in enumerate(nodes):
        for t in range(n_retrain):
            # Normal training
            Ev.train_model(surrogate=False, with_perturb=False, disp=False)
            margins[i, t, 0] = Ev.margins[node]
            # Normal attack
            Ev.attack(u=node, verbose=False)
            Ev.train_model(surrogate=False, with_perturb=True, disp=False)
            margins[i, t, 1] = Ev.margins[node]
            # Defending
            EvDef.defend(u=node, verbose=False, perturb_features=True)
            EvDef.train_model(surrogate=False, with_perturb=False, disp=False)
            margins[i, t, 2] = EvDef.margins[node]
            # Attacking defended graph
            margins[i, t, 3] = margin_attack(Ev, sparse_numpy2sparse_torch(EvDef.nettack.adj_preprocessed),
                                                sparse_numpy2sparse_torch(EvDef.nettack.X_obs))
            pbar.update(1)
    pbar.close()
    return margins, nodes