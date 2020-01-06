import tqdm
import sklearn
import sklearn.metrics
import math

import torch

from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F

spdot = torch.sparse.mm
dot = torch.matmul


# TODO: convert this to sparse torch tensors
def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1. / keep_prob)


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, add_bias=True):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None and add_bias:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    #TODO: set the device, GPU/CPU
    def __init__(self, sizes, An, X_obs, name="", with_relu=True, params_dict={},
                 seed=-1, bias=True):
        """
        Create a Graph Convolutional Network model in PyTorch with one hidden layer.
        Parameters
        ----------
        sizes: list
            List containing the hidden and output sizes (i.e. number of classes). E.g. [16, 7]
        An: torch.sparse matrix, shape [N,N]
            The input adjacency matrix preprocessed using the procedure described in the GCN paper.
        X_obs: torch.sparse matrix, shape [N,D]
            The node features.
        name: string, default: ""
            Name of the network.
        with_relu: bool, default: True
            Whether there a nonlinear activation function (ReLU) is used. If False, there will also be
            no bias terms, no regularization and no dropout.
        params_dict: dict
            Dictionary containing other model parameters.
        gpu_id: int or None, default: 0
            The GPU ID to be used by Pytorch. If None, CPU will be used
        seed: int, defualt: -1
            Random initialization for reproducibility. Will be ignored if it is -1.
        """
        super(GCN, self).__init__()
        if seed > -1:
            torch.manual_seed(seed)

        if not An.is_sparse:
            An = An.to_sparse()

        self.name = name

        self.n_hidden, self.n_classes = sizes

        self.dropout = params_dict['dropout'] if 'dropout' in params_dict else 0.
        if not with_relu:
            self.dropout = 0

        self.learning_rate = params_dict['learning_rate'] if 'learning_rate' in params_dict else 0.01

        self.weight_decay = params_dict['weight_decay'] if 'weight_decay' in params_dict else 5e-4
        self.N, self.D = X_obs.shape

        self.gc1 = GraphConvolution(self.D, self.n_hidden, bias)
        self.gc2 = GraphConvolution(self.n_hidden, self.n_classes, bias)

        self.An = An
        if not X_obs.is_sparse:
            self.X_sparse = X_obs.to_sparse()
        else:
            self.X_sparse = X_obs

        self.training = False
        self.with_relu = with_relu

    def forward(self, node_ids):
        """
        The forward pass is made on all nodes, but only nodes in node_ids have
        their logits returned.
        """
        # only use drop-out during training

        if self.dropout > 0 and self.training:
            self.X_comp = sparse_dropout(self.X_sparse, 1 - self.dropout,
                                         (int(self.X_sparse.shape[0]),))
        else:
            self.X_comp = self.X_sparse

        self.h1 = self.gc1(adj=self.An, input=self.X_comp)

        if self.with_relu:
            self.h1 = F.relu(self.h1)

        self.h1_dropout = F.dropout(self.h1, self.dropout, training=self.training)
        self.h1_comp = self.h1_dropout if self.dropout > 0. and self.training else self.h1

        self.logits = self.gc2(adj=self.An, input=self.h1_comp, add_bias=False)

        if self.with_relu:
            self.logits += self.gc2.bias

        logits_gather = self.logits[node_ids]
        return logits_gather


class GCN_Model():
    def __init__(self, gcn, lr=1e-3, path="models/"):
        self.gcn = gcn
        self.optimizer = torch.optim.Adam(self.gcn.parameters(), lr=lr)
        self.path = path

    def _compute_loss_and_backprop(self, node_ids, node_labels, backward=True):
        """
        Makes a forward and backward pass for node_ids.
        """
        self.gcn.training = backward
        self.logit_nodes = self.gcn(node_ids)
        self.predictions = F.softmax(self.logit_nodes, dim=1)
        self.loss_per_node = F.cross_entropy(input=self.logit_nodes, target=node_labels, reduction='none')
        self.loss = self.loss_per_node.mean()
        if self.gcn.with_relu:
            self.loss += self.gcn.weight_decay * sum([(x ** 2).sum()
                                                      for x in [self.gcn.gc1.weight, self.gcn.gc1.bias]])
        self.optimizer.zero_grad()
        if backward:
            self.loss.backward()
            self.optimizer.step()
        return self.loss

    def _predict(self, node_ids, train=False, name=None):
        """
        Used at the moment for the function eval_class. Can be used to produce
        only prediction probabilities.
        """
        # if name is not None:
            # self.gcn.load_state_dict(torch.load(self.path + f"{name}.pth"))
        if not train:
            self.gcn.training = False
            with torch.no_grad():
                logit_nodes = self.gcn(node_ids)
        else:
            self.gcn.training = True
            logit_nodes = self.gcn(node_ids)
        return F.softmax(logit_nodes, dim=1)

    def train(self, split_train, split_val, Z_obs, patience=30, n_iters=200, print_info=True, debug=False):
        """
        Train the GCN model on the provided data.
        Parameters
        ----------
        split_train: np.array, shape [n_train,]
            The indices of the nodes used for training
        split_val: np.array, shape [n_val,]
            The indices of the nodes used for validation.
        Z_obs: np.array, shape [N]
            All node labels in true encoding form (in 0, ..., C-1)
            (the labels of nodes outside of split_train and split_val will not be used.
        patience: int, default: 30
            After how many steps without improvement of validation error to stop training.
        n_iters: int, default: 200
            Maximum number of iterations (usually we hit the patience limit earlier)
        print_info: bool, default: True
        path: string, path to the folder where the weights of the model are to be saved
        Returns
        -------
        None.
        """
        early_stopping = patience

        best_performance = 0

        train_nodes = torch.tensor(split_train)
        train_labels = Z_obs[train_nodes]
        val_nodes = torch.tensor(split_val)
        val_labels = Z_obs[val_nodes]

        pbar = tqdm.tqdm_notebook(disable=(not debug))
        for it in range(n_iters):
            train_loss = self._compute_loss_and_backprop(train_nodes, train_labels)
            val_loss = self._compute_loss_and_backprop(val_nodes, val_labels, False)
            if debug:
                print("Iteration {} : ".format(it))
                print("Training loss : {}".format(train_loss))
                print("Validation loss : {}".format(val_loss))

            f1_micro, f1_macro = eval_class(split_val, self, Z_obs)
            perf_sum = f1_micro + f1_macro
            f1_micro_train, f1_macro_train = eval_class(split_train, self, Z_obs)
            perf_sum_train = f1_micro_train + f1_macro_train
            if debug:
                print("Training metric : {}".format(perf_sum_train))
                print("Validation metric : {}".format(perf_sum))

            if perf_sum > best_performance:
                # Save current best model
                best_performance = perf_sum
                best_it = it
                patience = early_stopping
                # torch.save(self.gcn.state_dict(), self.path + "weights.pth")
                if debug:
                    print(f'New best performance : {perf_sum:.3f}')
            else:
                patience -= 1
            if it > early_stopping and patience <= 0:
                pbar.close()
                if print_info:
                    print('converged after {} iterations'.format(best_it))
                break
            pbar.update(1)
            if it == n_iters - 1:
                pbar.close()
                if print_info:
                    print('converged after {} iterations'.format(best_it))


def eval_class(ids_to_eval, model, z_obs):
    """
    Evaluate the model's classification performance.
    Parameters
    ----------
    ids_to_eval: np.array
        The indices of the nodes whose predictions will be evaluated.
    model: GCN_model
        The model to evaluate.
    z_obs: 1d np.array
        The labels of the nodes in ids_to_eval
    Returns
    -------
    [f1_micro, f1_macro] scores
    """
    test_pred = model._predict(ids_to_eval).argmax(1)
    test_real = z_obs[ids_to_eval]
    return (sklearn.metrics.f1_score(test_real, test_pred, average='micro'),
            sklearn.metrics.f1_score(test_real, test_pred, average='macro'))
