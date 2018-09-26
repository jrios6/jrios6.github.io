---
layout: post
title:  "Summary of Residual Gated Graph ConvNets"
description: >
  X. Bresson and T. Laurent. [Residual Gated Graph ConvNets](https://arxiv.org/pdf/1711.07553.pdf). arXiv preprint arXiv:1711.07553.
categories: fyp
---
- Focus of this paper was to compare between RNN and ConvNet architectures on two basic graph problems: Subgraph Matching and Semi-Supervised Clustering, in the variable graph setting.

### Experimental Setup
- In **Subgraph Matching**, the goal is to find the vertices of a given subgraph P (20 Nodes) in larger graphs $$ G_{k} $$ of variable sizes (15-25 Nodes/ Community), with the same signals. During training and testing, 5000 and 100 random $$ G_{k} $$ with P are generated respectively.

- These graphs are generated using the Stochastic Block Model (SBM), where two nodes in a random graph are connected with probability p (=0.5) if they are from the same community and q (=0.1) if otherwise. Each node has a random signal of {0,1,2} and belongs to one of the 10 communities.

- In **Semi-Supervised Clustering**, we are given 1 single label for each of the 10 communities and the goal is to determine the community for each vertex. These graphs are generated with SBM, with each community having 5-25 nodes and p = 0.5. Each node has a random signal of {0,1,2...,10}. During training and testing, 5000 and 100 random graphs are generated respectively.

### Methods
- Authors proposed a multi-layer extension of the **Tree-LSTM** of Tai et al. (2015) for variable graphs:

$$
\begin{gathered}
  \kern{1em} &h_{i}^{l,t+1} = o_{i}^{l,t+1} \odot tanh(c_{i}^{l,t+1}),\\[0.5em]
  &\text{ where } h_{i}^{l,t=0} = c_{i}^{l,0} = 0            \\[0.5em]
  &x_{i}^{l} = h_{i}^{l-1,T}, x_{i}^{t=0} = x_{i} \text{, for all i, l}

\end{gathered}
$$

- Authors proposed a **multi-layer graph ConvNet with edge gating and residuality** for variable graphs, based on the vanilla Graph ConvNet architecutre of Sukhbaatar et al. (2016), Eq.(7), and the edge gating mechanism of Marcheggiani & Titov (2017).

- Authors used an **embedding layer** with 50 output dimension for each of the 10 possible vertex signals.

- Graphs are represented in the form of **incidence matrixes**, with an edge-to-starting-vertex matrix and an edge-to-ending-vertex matrix (# edges x # nodes).

- In training, each graph is shown only once to prevent overfitting.

- Dropout can be applied both on edges and nodes as regularization.


### Results
- Results show that ConvNets are **3-17%** more accurate and **1.5-4x** faster than Graph RNNs, and 36% more accurate than non-learning techniques.

- Gated edges and residuality provide a 10% gain in performance among ConvNet architectures.  


### Questions:
  **Q:** Why is the signal for each node generated randomly? How can this be helpful for the GNN to classify the community of each node?  
  **A:** Just to control the train/ test environment. Currently these signals are not really helpful.  

  **Q:** In Subgraph Matching, is the network trying to classify if a node in G is part of subgraph P or not? Or is it trying to classify the exact community each node in G and P belongs to?  
  **A:** The FC layers is doing binary classification between P class and G class.

  **Q:** In Semi-Supervised Clustering, embedding layer has input of 10+1 dimensions, but how many possible signals {0,1,2} are there for each node?  
  **A:** There should be the same number of signals as per the embedding dimensions.

  **Q:** How well would the multi-layer graph ConvNet perform on fixed graphs?  
  **A:** It should not perform that well as the spatial version of GNN requires multiple layers to propagate the graph structure forward. Whereas in Spectral ConvNet, Fourier analysis allows the entire graph structure to be captured by the ConvNet. For fixed graphs, spectral technique should work better. However, spectral technique won't work well for variable graphs as any changes to the graph structure would cause the Fourier analysis of the graph to change vastly.

  **Q:** For Subgraph Matching, loss function is cross-entropy between class of P and $$ G_{k} $$ weighted respectively by their size. How is the weightage computed?  
  **A:** Since the dataset is unbalanced (# of P-class $$ \ll $$ # of G-class), we need to reweigh the loss function. Can be as simple as multiply the loss of P-node by 1/(# of P's) and loss of G-node by 1/(# of G's).
