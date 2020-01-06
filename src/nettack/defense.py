"""
Functions for making a graph more robust
"""
import numpy as np
import scipy.sparse as sp
import tqdm

from .nettack import *
from .GCN import GCN, GCN_Model
from .replication import Evaluater, sparse_numpy2sparse_torch
from .utils import preprocess_graph

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



class NetRandomize(Nettack):

    def randomize(self, n_perturbations, perturb_structure=True, perturb_features=True,
                  direct=True, n_influencers=0, delta_cutoff=0.004):
        """
        Perform an attack on the surrogate model.
        Parameters
        ----------
        n_perturbations: int
            The number of perturbations (structure or feature) to perform.
        perturb_structure: bool, default: True
            Indicates whether the structure can be changed.
        perturb_features: bool, default: True
            Indicates whether the features can be changed.
        direct: bool, default: True
            indicates whether to directly modify edges/features of the node attacked or only those of influencers.
        n_influencers: int, default: 0
            Number of influencing nodes -- will be ignored if direct is True
        delta_cutoff: float
            The critical value for the likelihood ratio test of the power law distributions.
             See the Chi square distribution with one degree of freedom. Default value 0.004
             corresponds to a p-value of roughly 0.95.
        Returns
        -------
        None.
        """

        assert not (direct==False and n_influencers==0), "indirect mode requires at least one influencer node"
        assert n_perturbations > 0, "need at least one perturbation"
        assert perturb_features or perturb_structure, "either perturb_features or perturb_structure must be true"

        logits_start = self.compute_logits()
        best_wrong_class = self.strongest_wrong_class(logits_start)
        surrogate_losses = [logits_start[self.label_u] - logits_start[best_wrong_class]]

        if self.verbose:
            print("##### Starting attack #####")
            if perturb_structure and perturb_features:
                print("##### Attack node with ID {} using structure and feature perturbations #####".format(self.u))
            elif perturb_features:
                print("##### Attack only using feature perturbations #####")
            elif perturb_structure:
                print("##### Attack only using structure perturbations #####")
            if direct:
                print("##### Attacking the node directly #####")
            else:
                print("##### Attacking the node indirectly via {} influencer nodes #####".format(n_influencers))
            print("##### Performing {} perturbations #####".format(n_perturbations))

        if perturb_structure:

            # Setup starting values of the likelihood ratio test.
            degree_sequence_start = self.adj_orig.sum(0).A1
            current_degree_sequence = self.adj.sum(0).A1
            d_min = 2
            S_d_start = np.sum(np.log(degree_sequence_start[degree_sequence_start >= d_min]))
            current_S_d = np.sum(np.log(current_degree_sequence[current_degree_sequence >= d_min]))
            n_start = np.sum(degree_sequence_start >= d_min)
            current_n = np.sum(current_degree_sequence >= d_min)
            alpha_start = compute_alpha(n_start, S_d_start, d_min)
            log_likelihood_orig = compute_log_likelihood(n_start, alpha_start, S_d_start, d_min)

        if len(self.influencer_nodes) == 0:
            if not direct:
                # Choose influencer nodes
                infls, add_infls = self.get_attacker_nodes(n_influencers, add_additional_nodes=True)
                self.influencer_nodes= np.concatenate((infls, add_infls)).astype("int")
                # Potential edges are all edges from any attacker to any other node, except the respective
                # attacker itself or the node being attacked.
                self.potential_edges = np.row_stack([np.column_stack((np.tile(infl, self.N - 2),
                                                                 np.setdiff1d(np.arange(self.N),
                                                                              np.array([self.u,infl])))) for infl in
                                                     self.influencer_nodes])
                if self.verbose:
                    print("Influencer nodes: {}".format(self.influencer_nodes))
            else:
                # direct attack
                influencers = [self.u]
                self.potential_edges = np.column_stack((np.tile(self.u, self.N-1), np.setdiff1d(np.arange(self.N), self.u)))
                self.influencer_nodes = np.array(influencers)
        self.potential_edges = self.potential_edges.astype("int32")
        for _ in range(n_perturbations):
            if self.verbose:
                print("##### ...{}/{} perturbations ... #####".format(_+1, n_perturbations))
            if perturb_structure:

                # Do not consider edges that, if removed, result in singleton edges in the graph.
                singleton_filter = filter_singletons(self.potential_edges, self.adj)
                filtered_edges = self.potential_edges[singleton_filter]

                # Update the values for the power law likelihood ratio test.
                deltas = 2 * (1 - self.adj[tuple(filtered_edges.T)].toarray()[0] )- 1
                d_edges_old = current_degree_sequence[filtered_edges]
                d_edges_new = current_degree_sequence[filtered_edges] + deltas[:, None]
                new_S_d, new_n = update_Sx(current_S_d, current_n, d_edges_old, d_edges_new, d_min)
                new_alphas = compute_alpha(new_n, new_S_d, d_min)
                new_ll = compute_log_likelihood(new_n, new_alphas, new_S_d, d_min)
                alphas_combined = compute_alpha(new_n + n_start, new_S_d + S_d_start, d_min)
                new_ll_combined = compute_log_likelihood(new_n + n_start, alphas_combined, new_S_d + S_d_start, d_min)
                new_ratios = -2 * new_ll_combined + 2 * (new_ll + log_likelihood_orig)

                # Do not consider edges that, if added/removed, would lead to a violation of the
                # likelihood ration Chi_square cutoff value.
                powerlaw_filter = filter_chisquare(new_ratios, delta_cutoff)
                filtered_edges_final = filtered_edges[powerlaw_filter]

                # Compute new entries in A_hat_square_uv
                a_hat_uv_new = self.compute_new_a_hat_uv(filtered_edges_final)
                # Compute the struct scores for each potential edge
                # struct_scores = self.struct_score(a_hat_uv_new, self.compute_XW())
                best_edge_ix = np.random.randint(a_hat_uv_new.shape[0])
                # best_edge_score = struct_scores[best_edge_ix]
                best_edge = filtered_edges_final[best_edge_ix]

            if perturb_features:
                # Compute the feature scores for each potential feature perturbation
                feature_ixs, feature_scores = self.feature_scores()
                random_feat = np.random.randint(len(feature_scores))
                best_feature_ix = feature_ixs[random_feat]
                best_feature_score = feature_scores[random_feat]

            if perturb_structure and perturb_features:
                # decide whether to choose an edge or feature to change
                if np.random.rand() < 0.5:
                    if self.verbose:
                        print("Edge perturbation: {}".format(best_edge))
                    change_structure = True
                else:
                    if self.verbose:
                        print("Feature perturbation: {}".format(best_feature_ix))
                    change_structure=False
            elif perturb_structure:
                change_structure = True
            elif perturb_features:
                change_structure = False

            if change_structure:
                # perform edge perturbation

                self.adj[tuple(best_edge)] = self.adj[tuple(best_edge[::-1])] = 1 - self.adj[tuple(best_edge)]
                self.adj_preprocessed = preprocess_graph(self.adj)

                self.structure_perturbations.append(tuple(best_edge))
                self.feature_perturbations.append(())
                # surrogate_losses.append(best_edge_score)

                # Update likelihood ratio test values
                current_S_d = new_S_d[powerlaw_filter][best_edge_ix]
                current_n = new_n[powerlaw_filter][best_edge_ix]
                current_degree_sequence[best_edge] += deltas[powerlaw_filter][best_edge_ix]

            else:
                self.X_obs[tuple(best_feature_ix)] = 1 - self.X_obs[tuple(best_feature_ix)]

                self.feature_perturbations.append(tuple(best_feature_ix))
                self.structure_perturbations.append(())
                surrogate_losses.append(best_feature_score)


class EvaluaterRand(Evaluater):

    def load_attacked_graph(self, Ev):
        self._A_obs = Ev.nettack.adj
        self._X_obs = Ev.nettack.X_obs
        self.A = sparse_numpy2sparse_torch(self._A_obs)
        self.An = sparse_numpy2sparse_torch(Ev.nettack.adj_preprocessed)
        self.X = sparse_numpy2sparse_torch(Ev.nettack.X_obs)
    
    def defend(self, u, n=10, verbose=False, n_perturbations=None, direct_attack=True,
               perturb_features=True, perturb_structure=True):
        self.ngraphs = n
        self.nettack = NetRandomize(self._A_obs, self._X_obs, self.Z, self.W1, self.W2, u, verbose=verbose)
        self.nettack.reset()
        if n_perturbations is None:
            n_perturbations = int(self.degrees[u])
        n_influencers = 1 if direct_attack else 5
        self.As = []
        self.Xs = []
        # Creating random adjacency and features matrices
        for _ in tqdm.tqdm_notebook(range(n), disable=(not verbose)):
            self.nettack.randomize(n_perturbations,
                            perturb_structure=perturb_structure,
                            perturb_features=perturb_features,
                            direct=direct_attack,
                            n_influencers=n_influencers)
            self.As.append(sparse_numpy2sparse_torch(self.nettack.adj_preprocessed))
            self.Xs.append(sparse_numpy2sparse_torch(self.nettack.X_obs))

    def add_defender_graph(self, u, verbose=False, n_perturbations=None, direct_attack=True,
               perturb_features=True, perturb_structure=True):
        self.ngraphs += 1
        self.nettack = NetRandomize(self._A_obs, self._X_obs, self.Z, self.W1, self.W2, u, verbose=verbose)
        self.nettack.reset()
        if n_perturbations is None:
            n_perturbations = int(self.degrees[u])
        n_influencers = 1 if direct_attack else 5
        self.nettack.randomize(n_perturbations,
                        perturb_structure=perturb_structure,
                        perturb_features=perturb_features,
                        direct=direct_attack,
                        n_influencers=n_influencers)
        self.As.append(sparse_numpy2sparse_torch(self.nettack.adj_preprocessed))
        self.Xs.append(sparse_numpy2sparse_torch(self.nettack.X_obs)) 

    def update_margins(self, disp=False):
        # Currently : uniform aggregation :
        self.logits_mean = np.atleast_2d(np.array(self.logits_models)).mean(0)
        self.preds = self.logits_mean.argmax(axis=1)
        truth = self.Z[self.split_val]
        if disp:
            print(f'Validation accuracy : {(self.Z[self.split_val] == self.preds[self.split_val]).mean():.2%}')
            print(f'Train accuracy : {(self.Z[self.split_train] == self.preds[self.split_train]).mean():.2%}')
            print(f'Unlabeled accuracy : {(self.Z[self.split_unlabeled] == self.preds[self.split_unlabeled]).mean():.2%}')
        self.probas = np.exp(self.logits_mean) / np.exp(self.logits_mean).sum(1)[:, None]
        probas_surr_sorted = np.argsort(-self.probas, axis=1)
        second_l = probas_surr_sorted[np.arange(self.N),
                                      (probas_surr_sorted == self.Z[:, None]).argmin(axis=1)]
        self.margins = (self.probas[np.arange(self.N), self.Z] -
                        self.probas[np.arange(self.N), second_l])

    def train_single_model(self, A, X, disp=True, debug=False):
        if not hasattr(self, 'logits_models'):
            self.logits_models = []
            self.margins_models = []
        sizes = [16, self.K]
        self.nn = GCN(sizes, A, X, with_relu=True)
        self.model = GCN_Model(self.nn, lr=1e-2)
        self.model.train(self.split_train, self.split_val, self.Ztorch, print_info=disp, debug=debug)
        
        # Computing logits for every node
        self.model._compute_loss_and_backprop(np.arange(self.N), self.Ztorch, backward=False)
        self.logits_model = self.model.logit_nodes.detach().cpu().numpy()
        preds = self.logits_model.argmax(axis=1)
        truth = self.Z[self.split_val]
        if disp:
            print(f'Validation accuracy : {(self.Z[self.split_val] == preds[self.split_val]).mean():.2%}')
            print(f'Train accuracy : {(self.Z[self.split_train] == preds[self.split_train]).mean():.2%}')
            print(f'Unlabeled accuracy : {(self.Z[self.split_unlabeled] == preds[self.split_unlabeled]).mean():.2%}')
        self.probas_model = np.exp(self.logits_model) / np.exp(self.logits_model).sum(1)[:, None]
        probas_surr_sorted = np.argsort(-self.probas_model, axis=1)
        second_l = probas_surr_sorted[np.arange(self.N),
                                      (probas_surr_sorted == self.Z[:, None]).argmin(axis=1)]
        self.margins_model = (self.probas[np.arange(self.N), self.Z] -
                        self.probas[np.arange(self.N), second_l])
        self.logits_models.append(self.logits_model)
        self.margins_models.append(self.margins_model)
        self.update_margins(disp=disp)
        

    def train_models(self, disp=True, debug=False):
        self.logits_models = []
        self.margins_models = []
        for i in tqdm.tqdm_notebook(range(self.ngraphs), disable=(not disp)):
            self.train_single_model(self.As[i], self.Xs[i], disp=disp, debug=debug)
            self.logits_models.append(self.logits_model)
            self.margins_models.append(self.margins_model)
        self.update_margins(disp=disp)
