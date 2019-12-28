# GraphsML2019
Repo for GRAPHS in ML 2019 project, MVA.
http://researchers.lille.inria.fr/~valko/hp/mvaprojects.php

Description:  Graph neural networks have been used in many application domains and is now one of the standard tools
for modeling and making predictions on graph structured data.  One intriguing property about neural networks is
their vulnerability against adversarial examples [1]; graph neural networks suffer from similar issues [2,3].
In addition to changing the input features, a unique attack against GNNs is the structural attack, i.e. changing
a small set of (or even a single) the edges / nodes in the graph may change the output of the GNN by a lot.
In the literature there has been some recent work developing attacks for GNNs [2,3], but the types of known attacks
are still limited.  On the other hand, there are even fewer work on defence against adversarial attacks on GNNs
(example: [7]).  Verifying GNNs (in a way like e.g. [4,5,6]) is another topic that presents unique challenges and
is under-studied.  This project involves (1) implementing some of the GNN attacks developed in the literature and
(2) do at least one of the following:

develop new or more efficient attacks

develop new defence or training algorithms that are robust against the above attacks

apply / develop neural network verification techniques for GNNs and handle the unique challenges
involving graph structure

[1] Intriguing properties of neural networks.

[2] Adversarial Attacks on Neural Networks for Graph Data.

[3] Adversarial Attack on Graph Structured Data

[4] A Dual Approach to Scalable Verification of Deep Networks

[5] Branch and Bound for Piecewise Linear Neural Network Verification

[6] On the Effectiveness of Interval Bound Propagation for Training Verifiably Robust Models

[7] Certifiable Robustness to Graph Perturbations



Implementation of Nettack : https://github.com/danielzuegner/nettack in PyTorch
