# R Software

## Clustering

Basic R functions for clustering:
- hclust, plot.hclust, cutree for hierarchical methods.
- kmeans for K-means.
- pam (package cluster) for K-medoids.
- Package ClusterR includes K-Means, K-Medoids and Gaussian

Mixture Models (GMM).
- How many clusters are there in a dataset?
- Gap statistic. See the function clusGap of the R package cluster.
- Silhouette. See the function silhouette of the R package cluster.
- Calinski-Harabasz index. See the function cluster.stats of the R package fpc.
- Package ClusterR includes functions that determine the optimal number of clusters in K-Means, K-Medoids and GMM.
- See the R package fpc for other methods.

## GMM

Package ClusterR:
- Function GMM fits Gaussian Mixture Models using the EM algorithm (with an inner K-means step).
- This library includes the function Optimal Clusters GMM to determine the optimal number of clusters in GMM.
- ClusterR also performs other clustering methods as K-Means and K-Medoids.

Package mclust: An R package for model-based clustering, classification, and density estimation based on finite normal mixture modeling.
- Function Mclust. Model-based clustering based on parameterized finite Gaussian mixture models. Models are estimated by EM algorithm. The optimal model is selected according to BIC (Bayesian Information Criterion).

Package fpc, Flexible Procedures for Clustering.
- mergenormals: Clustering by merging components of a Gaussian mixture.
- cluster.stats: Computes several cluster validity statistics from a clustering and a dissimilarity matrix.

## DBSCAN

Package fpc, Flexible Procedures for Clustering.
- dbscan: Computes DBSCAN density based clustering as introduced in Ester, Kriegel, Sander, Xu, et al. (1996).
- cluster.stats: Computes several cluster validity statistics from a clustering and a dissimilarity matrix.
Package dbscan, function dbscan: Fast reimplementation of the DBSCAN.