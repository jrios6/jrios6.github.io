---
layout: post
title:  "FYP Project Updates"
description: > Semi-Supervised Clustering Project Progress Updates
categories: fyp
---

# FYP Project Updates

## Update #8 (01/03/19)
- For reference, with GCN: Citeseer: 70.3%, Pudmed: 79.0%
- Experimental Results:
- Citeseer - 62.8% Val. Acc.
- >4% improvement with the use of random edge additions
- Unable to load Pudmed to memory for now as edge-to-start matrix takes 109k X 19k space, working on sparse matrix implementation/ increase RAM on Cloud VM.

## Update #7 (15/02/19)
### Modifications to RGGCN
- Removed $$U^{l}H_{i}^{l}$$ term in RGGCN. Feature vector of the $$H_i$$ is added for $$H_{i+1}$$ via self-edges in the second term. This reduces the number of learnable parameters by 25%, and introduces the edge gating property to the self-edge.
- Added edge normalization to edge gates, normalizing each edge gate $$n_{ij}$$ by the sum of edges from $$j \rightarrow i$$. This is similar to the normalization of attention coefficients. From my experiments, edge normalization improves learning speed of the network, allowing training to converge more quickly on deeper networks. Dropout is not applied on self-edges.
- Replaced Relu with Leaky Relu in RGGCN.
- Updated learning rate decay to decrease LR by 20% if avg. val loss in last 20 epochs is higher than the previous 20 epochs.
- Added early stopping criteria when learning rate decreases to 1/4 of initial and test acc. is at least 5% higher than val acc.
- With these improvements, RGGCN managed to achieve 80.4% accuracy on pyGAT Cora Dataset. In the pyGAT dataset, 140 training nodes were selected proportionally from the 7 classes, rather than 20 nodes per class.
- After switching the datasets to the original one used in GCN, the best validation accuracy decreased to 78.8%. This might be due to overfitting on the initial validation set, or the proportional distribution of training nodes made a significant difference.
- Wrote a script for random search of hyperparameters for Dropout Edge, Dropout FC, Random Edge Noise and L2.

### Graph Regularization
- Overfitting remains an issue despite the use of Dropout and Early Stopping. There are few regularization solutions on hand for GNNs since it is not possible to apply traditional data augmentation on a fixed graph or add gaussian noise to the input feature vector, which is sparse vector representing bag-of-words.
- One solution is to add noise to the input feature vector at random. Since it represents a bag-of-words, we can randomly set a word to 1 for x% of documents. The is the inverse of dropout, which randomly removes x% of words from each document.
- Another solution is to add an edge between two nodes at random. This introduces noise in training, forcing the network to learn to be more robust to noisy edge connections present in graph data. In practice, even addition of <1% of possible edges (19000 in total) to the graph is sufficient to prevent overfitting in training. However, improvements to generalization error seems to be limited.
- Another solution is to integrate more parameter sharing by combining weight matrix **A** and **B** used in learning the edge gates to a single weight matrix. However in practice, the network performed poorly with parameter sharing of **A** and **B**.

### Adversarial Training and Semi-Supervised Learning
- Adversarial training is a method that intentionally introduces perturbations to the input to maximise the error rate of the classifier. Neural networks are made up of linear building blocks, which can produce highly different outputs with a small change to the input. Adversarial training encourages the network to be locally consistent in the neighbourhood of the training data. This technique, extended to graphs, would encourage the network to learn a function that is robust to small changes to features of a node or its neighbourhood.
- In semi-supervised classification, Virtual Adversarial Training [[Miyato et. al., 2018](https://arxiv.org/pdf/1704.03976.pdf)] can be applied by seeking adversarial samples that causes the network to predict a different label even for unlabelled nodes.
- Lastly, we move towards a general semi-supervised learning by combining the supervised loss we have now, with an unsupervised loss for the unlabelled nodes. This requires a corruption function to modify our fixed graph and adding a 'fake' class to our output classifier. The loss function seeks to maximise the probability of classifying the correct class on labelled nodes and a fake node as a 'fake' class, and minimize the probability of classifying a unlabelled node as a fake node. The core difference in different techniques in this domain, is the choice of the corruption function and the discriminator.
- Deep Graph Infomax [[Velickovic et. al., 2019](https://arxiv.org/abs/1809.10341)], a new unsupervised learning method that relies on maximising mutual information between node-level representations and high-level summary of graphs, provides a promising new approach for semi-supervised learning. This technique will be explored in a more detailed blog post.
- Some of the current experiments I have with DGI involves the addition of a supervised loss to the loss function. The hypothesis is that semi-supervised learning allows the GCN layers to directly learn representations that are more useful for the classification task, rather than rely on the logistic regression layer to map the generic embeddings with the output classes.
- Currently, I am able to attain 80% test accuracy using a 2-layer GCN, with the main challenge being weighing the two losses, and regularization to prevent overfitting on the labelled nodes.

### References
[Goodfellow et. al. Chapter 7, Regularization for Deep Learning. _In Deep Learning._](https://www.deeplearningbook.org/contents/regularization.html)  

[Miyato et. al. Virtual Adversarial Training: A Regularization Method for Supervised and Semi-Supervised Learning. _In IEEE transactions on pattern analysis and machine intelligence (TPAMI)_, 2018.](https://arxiv.org/pdf/1704.03976.pdf)  

[Velickovic et. al. Deep Graph InfoMax. _In ICLR_, 2019](https://arxiv.org/abs/1809.10341)  


## Update #6 (31/01/19)
- Uploaded project notebooks to [Github repository](https://github.com/jrios6/graph-neural-networks)
- Improved performance of RGGCN on Cora Dataset by using Adjacency + Identity Matrix for input
- Refactored RGGCN on Cora for faster training speed, and fixed bug in accuracy computation
- Re-experimented all-edge dropout, convolution block edge dropout, edge gating dropout, convolution output dropout and pre fully-connected layer dropout on RGGCN
- Some of the test accuracies (averaged of two runs) are reported below:

| Layers   | Dropout                                             | Test Acc.   |
| -------- | --------------------------------------------------- | ----------- |
| 4        | all-edge (0.3)                                      | 66.6%       |
| 4        | all edge dropout (0.3)                              | 65.5%       |
| 4        | fc dropout (0.3)                                    | 53.9%       |
| 4        | conv block edge dropout (0.3)                       | 69.5%       |
| 4        | conv block edge dropout, fc dropout (0.3)           | 63.5%       |
| 4        | conv output dropout (0.3)                           | 63.5%       |
| 4        | conv block edge dropout, conv output dropout (0.3)  | 68.2%       |
| **6**    | **all edge dropout, conv output dropout (0.3)**     | **70.9%**   |
| 6        | conv block edge dropout, conv output dropout (0.3)  | 69.8%       |
| 8        | all edge dropout, conv output dropout (0.3)         | 69.2%       |

- Some observations are that all edge dropout outperforms conv block dropout as number of layers increases. (All edge dropout refers to dropping out edges at the start of the feed-forward phase. Conv block dropout refers to dropping out edges in each conv block, with 3 conv blocks for a 6-layer RGGCN.)
- Without any form of dropout, overfitting occurs very quickly in training.
- The best combination of dropout currently is by combining all edge dropout with dropout of the output from each graph convolution layer, achieving >70% test accuracy (~10% lower than GCN/GAT).

## Update #5 (26/11/18)
- Experimented RGGCN with a color classification dataset (Google-512) where the task is to classify an input image among 11 colors.
- Chose this task as CNN and transfer learning does not work as well as other non-deep learning methods like LDA and KNN.
- Tested RGGCN with input nearest neighbour graph for each pixel based on similarity to neighbouring pixels, and added fully connected layer after to output 11 class predictions.
- Test performance was below that of a standard MLP, transfer learning with inception and a standard ConvNet

## Update #4 (12/11/18)
- Replaced final output attention layer on pyGAT with fully connected, similar to Residual Gated Graph ConvNet
- Improved performance for multi-layer attention, achieving 71% best test accuracy with 7 layers and 8 attention heads for Semi-Supervised Clustering

## Update #3 (15/10/18)
- Ran pyGAT on Semi-Supervised Clustering with Stochastic Block Model Dataset, achieving 62% best test accuracy
- Experimented with increasing neighbourhood size, assigning constant importance like Const-GAT
- Experimented with adding residuality to pyGAT, achieving 67% best test accuracy, with 4 layers, 8 attention heads and 4 output heads

## Update #2 (16/09/18)
- Setup Virtual Machine with K80 GPU on Google Cloud
- Read Graph Attention Networks (GATs)
- Wrote a Python script for converting Stochastic Block Model dataset into normalized adjacency
matrix for input to GATs
- Added signal embedding layer to pyGAT code to support SBM dataset
- Modify pyGAT negative log likelihood function to incorporate weight
- Added learning rate decay to pyGAT
- Wrote a function to compute accuracy on SBM testset for pyGAT model
- Experimented with different number of features per layer. Best test accuracy ~62%

## Update #1 (03/09/18)
- Setup Github Blog for posting weekly FYP progress reports and blog posts related to Graph Neural Networks.
- Wrote an introductory post with an overview of the Final Year Project, specifically on the task of semi-supervised clustering on graphs.
- Prepared FYP Project Plan detailing project objectives and milestones over the upcoming two semesters.
- Reviewed and wrote a paper summary for Residual Gated Graph ConvNets
