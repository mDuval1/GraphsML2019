import numpy as np
from sklearn.metrics import f1_score
import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F

# spdot = tf.sparse_tensor_dense_matmul
# dot = tf.matmul
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


class GraphConvolution(Module):
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


# TODO: review if init is ok
class GCN(nn.Module):
    def __init__(self, sizes, An, X_obs, name="", with_relu=True, params_dict={'dropout': 0.5}, gpu_id=0,
                 seed=-1, bias=True):
        """
        Create a Graph Convolutional Network model in PyTorch with one hidden layer.
        Parameters
        ----------
        sizes: list
            List containing the hidden and output sizes (i.e. number of classes). E.g. [16, 7]
        An: sp.sparse_matrix, shape [N,N]
            The input adjacency matrix preprocessed using the procedure described in the GCN paper.
        X_obs: sp.sparse_matrix, shape [N,D]
            The node features.
        name: string, default: ""
            Name of the network.
        with_relu: bool, default: True
            Whether there a nonlinear activation function (ReLU) is used. If False, there will also be
            no bias terms, no regularization and no dropout.
        params_dict: dict
            Dictionary containing other model parameters.
        gpu_id: int or None, default: 0
            The GPU ID to be used by Tensorflow. If None, CPU will be used
        seed: int, defualt: -1
            Random initialization for reproducibility. Will be ignored if it is -1.
        """
        super(GCN, self).__init__()
        if seed > -1:
            torch.manual_seed(seed)

        if An.format != "csr":
            An = An.tocsr()

        #need to set the device, cpu or gpu

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

        self.An = torch.sparse.FloatTensor(np.array(An.nonzero()).T, An[An.nonzero()].A1, An.shape).to_dense()
        self.X_sparse = torch.sparse.FloatTensor(np.array(X_obs.nonzero()).T, X_obs[X_obs.nonzero()].A1,
                                                 X_obs.shape).to_dense()

    #TODO: review if forward pass is ok, maybe don't compute the loss here
    def forward(self, node_ids, node_labels, training, with_relu=True):
        # only use drop-out during training
        self.X_dropout = sparse_dropout(self.X_sparse, 1 - self.dropout,
                                        (int(self.X_sparse.shape[0]),))
        self.X_comp = self.X_dropout if self.dropout > 0. and training else self.X_sparse

        self.h1 = self.gc1(self.An, self.X_comp)

        if with_relu:
            self.h1 = F.relu(self.h1)

        self.h1_dropout = F.dropout(self.h1, 1 - self.dropout, training=training)
        self.h1_comp = self.h1_dropout if self.dropout > 0. and training else self.h1

        self.logits = self.gc2(self.An, self.h1_comp, False)

        if with_relu:
            self.logits += self.gc2.bias

        self.logits_gather = torch.gather(self.logits, node_ids)
        self.predictions = F.softmax(self.logits_gather)

        self.loss_per_node = F.softmax_cross_entropy_with_logits(logits=self.logits_gather,
                                                                 labels=node_labels)
        self.loss = np.mean(self.loss_per_node)

        # weight decay only on the first layer, to match the original implementation
        # regularisation
        if with_relu:
            self.loss += self.weight_decay * sum([F.l2_loss(v) for v in [self.gc1.weight, self.gc1.bias]])

        var_l = [self.gc1.weight, self.gc2.weight]
        if with_relu:
            var_l.extend([self.gc1.bias, self.gc2.bias])

    #TODO: convert the training phase to PyTorch
    def train(self, split_train, split_val, Z_obs, patience=30, n_iters=200, print_info=True):
        """
        Train the GCN model on the provided data.
        Parameters
        ----------
        split_train: np.array, shape [n_train,]
            The indices of the nodes used for training
        split_val: np.array, shape [n_val,]
            The indices of the nodes used for validation.
        Z_obs: np.array, shape [N,k]
            All node labels in one-hot form (the labels of nodes outside of split_train and split_val will not be used.
        patience: int, default: 30
            After how many steps without improvement of validation error to stop training.
        n_iters: int, default: 200
            Maximum number of iterations (usually we hit the patience limit earlier)
        print_info: bool, default: True
        Returns
        -------
        None.
        """

        early_stopping = patience

        best_performance = 0

        feed = {self.node_ids: split_train,
                self.node_labels: Z_obs[split_train]}
        if hasattr(self, 'training'):
            feed[self.training] = True
        for it in range(n_iters):
            torch.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss, var_list=var_l)
            f1_micro, f1_macro = eval_class(split_val, self, np.argmax(Z_obs, 1))
            perf_sum = f1_micro + f1_macro
            if perf_sum > best_performance:
                best_performance = perf_sum
                patience = early_stopping
                # var dump to memory is much faster than to disk using checkpoints
                var_dump_best = {v.name: v.eval(self.session) for v in varlist}
            else:
                patience -= 1
            if it > early_stopping and patience <= 0:
                break
        if print_info:
            print('converged after {} iterations'.format(it - patience))
        # Put the best observed parameters back into the model
        self.set_variables(var_dump_best)

# TODO: I think the test_pred line is not correct
def eval_class(ids_to_eval, model, z_obs):
    """
    Evaluate the model's classification performance.
    Parameters
    ----------
    ids_to_eval: np.array
        The indices of the nodes whose predictions will be evaluated.
    model: GCN
        The model to evaluate.
    z_obs: np.array
        The labels of the nodes in ids_to_eval
    Returns
    -------
    [f1_micro, f1_macro] scores
    """
    test_pred = model.predictions.gather(ids_to_eval).argmax(1)
    test_real = z_obs[ids_to_eval]

    return f1_score(test_real, test_pred, average='micro'), f1_score(test_real, test_pred, average='macro')
