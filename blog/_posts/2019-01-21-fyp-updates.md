---
layout: post
title:  "FYP Project Updates"
description: > Semi-Supervised Clustering Project Progress Updates
categories: fyp
---

# FYP Project Updates

### Update #6 (31/01/19)
- Uploaded project notebooks to [Github repository](https://github.com/jrios6/graph-neural-networks)
- Improved performance of RGGCN on Cora Dataset by using Adjacency + Identity Matrix for input
- Refactored RGGCN on Cora for faster training speed, and fixed bug in accuracy computation

### Update #5 (26/11/18)
- Experimented RGGCN with a color classification dataset (Google-512) where the task is to classify an input image among 11 colors.
- Chose this task as CNN and transfer learning does not work as well as other non-deep learning methods like LDA and KNN.
- Tested RGGCN with input nearest neighbour graph for each pixel based on similarity to neighbouring pixels, and added fully connected layer after to output 11 class predictions.
- Test performance was below that of a standard MLP, transfer learning with inception and a standard ConvNet

### Update #4 (12/11/18)
- Replaced final output attention layer on pyGAT with fully connected, similar to Residual Gated Graph ConvNet
- Improved performance for multi-layer attention, achieving 71% best test accuracy with 7 layers and 8 attention heads for Semi-Supervised Clustering

### Update #3 (15/10/18)
- Ran pyGAT on Semi-Supervised Clustering with Stochastic Block Model Dataset, achieving 62% best test accuracy
- Experimented with increasing neighbourhood size, assigning constant importance like Const-GAT
- Experimented with adding residuality to pyGAT, achieving 67% best test accuracy, with 4 layers, 8 attention heads and 4 output heads

### Update #2 (16/09/18)
- Setup Virtual Machine with K80 GPU on Google Cloud
- Read Graph Attention Networks (GATs)
- Wrote a Python script for converting Stochastic Block Model dataset into normalized adjacency
matrix for input to GATs
- Added signal embedding layer to pyGAT code to support SBM dataset
- Modify pyGAT negative log likelihood function to incorporate weight
- Added learning rate decay to pyGAT
- Wrote a function to compute accuracy on SBM testset for pyGAT model
- Experimented with different number of features per layer. Best test accuracy ~62%

### Update #1 (03/09/18)
- Setup Github Blog for posting weekly FYP progress reports and blog posts related to Graph Neural Networks.
- Wrote an introductory post with an overview of the Final Year Project, specifically on the task of semi-supervised clustering on graphs.
- Prepared FYP Project Plan detailing project objectives and milestones over the upcoming two semesters.
- Reviewed and wrote a paper summary for Residual Gated Graph ConvNets
