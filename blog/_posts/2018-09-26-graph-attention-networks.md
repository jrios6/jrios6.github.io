---
layout: post
title:  "Summary of Graph Attention Networks"
description: >
  P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò and Y. Bengio. [Graph Attention Networks](https://openreview.net/pdf?id=rJXMpikCZ). ICLR 2018.  
categories: fyp
---

- This paper introduces Graph Attention Networks (GATs), a novel neural network architecture based on masked self-attention layers for graph-structured data.

- A Graph Attention Network is composed of multiple Graph Attention and Dropout layers, followed by a softmax or a logistic sigmoid function for single/multi-label classification.


### Graph Attention Layer
- A single Graph Attention Layer is parameterized by $$a \, \epsilon\ R^{2F'}$$  and $$W \,\epsilon\ R^{F' x F}$$, with input $$h \,\epsilon\, R^{N x F}$$ ($$N$$ nodes and $$F$$ features per node) and output $$h' \,\epsilon\, R^{N x F'}$$.

- First, **self-attention coefficient** $$e_{ij} = $$ LeakyReLu$$({a}^T [W\vec{h_i}\| W\vec{h_j}^T])$$ is computed, with $$\|$$ representing concatenation and LeakyReLu set with $$\alpha$$ of 0.2.

- $$e_{ij}$$ is the attention coefficient between node $$i$$ and node $$j$$, which indicates the importance of node $$j$$'s features to node $$i$$.

- **Masked attention** is employed to maintain the graph structure - by computing $$e_{ij}$$ only for nodes $$j \, \epsilon\ N_i$$, where $$N_i$$ is some *neighbourhood* of node $$i$$ in the graph. In this paper, the neighbourhood size is fixed to 1 (and includes node $$i$$ itself).

- The coefficients are then **normalized** (across all choices of $$j$$) using the following **softmax function** to make them easily comparable across different nodes:
$$
\alpha_{i,j} = softmax_j \, (e_{ij}) = \frac {exp(e_{ij})} {\sum_{k \, \epsilon\ N_i} exp(e_{ik})}
$$

- The final output feature for node $$i$$ is then computed as:
$$
\vec{h_i'} = \sigma(\sum_{j \epsilon N_i} \alpha_{i,j} W\vec{h_j})
$$
where $$\sigma$$ is the activation function.

- Multi-head attention is used to stabilize the learning process of self-attention, by **concatenating** the output of **K independent attention** mechanisms:
$$
\vec{h_i'} = ||_{k=1}^K \sigma(\sum_{j \epsilon N_i} \alpha_{i,j}^k W^k\vec{h_j})
$$
where $$\alpha_{i,j}^k$$ is the normalized attention coefficient computed with the $$k$$-th attention mechanism $$a^k$$ and $$W^k$$ is the corresponding weight matrix. Hence, the output $$\vec{h_i}$$ will have $$KF'$$ instead of $$F'$$ features.

- For the final (prediction) layer of the network, **averaging** is first applied to the sum of the output from each attention head, before the final non-linearity is applied:
$$
\vec{h_i'} = \sigma(\frac{1}{K} \sum_{k=1}^K \sum_{j \epsilon N_i} \alpha_{i,j}^k W^k\vec{h_j})
$$

- During training, **dropout** is applied to layer's **input** and normalized attention coefficients, **$$\alpha_{ij}$$**. Hence, each node is exposed to a stochastically sampled neighbourhood.

### Advantages of GATs
- A single GAT attention head with $$F'$$ features can be computed in $$O(\text{\textbar}V\text{\textbar}FF' + \text{\textbar}E\text{\textbar}F')$$ , on par with Graph Convolutional Networks (GCNs) (Kipf & Welling, 2017)

- Unlike GCNs, GATs allows for (implicit) assigning of different importances to nodes in the same neighbourhood. Analyzing the weights might lead to benefits in interpretability.

- The attention mechanism is applied in a shared manner across all edges in the graph, hence it does not require upfront access to the global graph strcture or features of all of its nodes.

- The graph is not required to be undirected. $$\alpha_{ij}$$ can be left out if edge $$j \rightarrow i$$ is not present.

- GAT can be used for both inductive (evaluated on graphs completely unseen during training) and transductive learning.

### Experimental Setup
- Transductive Learning (**Cora & Citeseer**): Two-layer GAT model, with $$K$$ = 8 attention heads and $$F'$$ = 8 features in first layer, followed by an exponential linear unit (ELU). Second layer is used for classfication with $$C$$ output features (# of classes), $$K$$ = 1, followed by a softmax activation. $$L_2$$ regularization is set to 0.0005. Dropout with $$p$$ = 0.6 is applied to both layers' inputs and normalized attention coefficients, $$\alpha_{ij}$$.  

- Transductive Learning (**Pubmed**): Two-layer GAT model, with $$K$$ = 8 and $$F'$$ = 8 in first layer, followed by ELU nonlinearity. Second layer is used for classfication with $$C$$ output features and $$K$$ = 8, followed by a softmax activation. $$L_2$$ regularization is set to 0.001 and dropout is set to $$p$$ = 0.6.  

- Inductive Learning (**PPI**): Three-layer GAT model, with $$K$$ = 4 and $$F'$$ = 256 features on first two layers, followed by ELU nonlinearity. Final layer has $$K$$ = 6 and $$C$$ = 121, followed by a logistic sigmoid activation. No regularization is used as the training set is sufficiently large. Skip connections are added across the intermediate attentional layer.

### Results
- GATs outperformed GCNs on Cora and Citeseer by 1.5% and 1.6% respectively, and matching GCNs performance on the Pubmed dataset.

- GATs outperformed GraphSAGE by 20.5% on the PPI dataset.


### Questions
**Q**: How would changing the neighbourhood size affect results? How about using different neighbourhood size in different layers?

**Q**: In the context of semi-supervised clustering, if signal embedding is used as input $$h_i$$, and majority of the unlabelled nodes sharing the same $$h_i$$, how can the attention co-efficient be effective computed?
