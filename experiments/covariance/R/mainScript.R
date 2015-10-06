# --- Clear workspace ---
rm(list=ls())

# --- Load packages ---
library('plyr')         # Data wrangling/munging
library('QUIC')         # Quadratic Graphical Lasso algorithm
library('PMA')          # Sparse SVD
library('corpcor')      # Stein-type shrinkage estimator
library('matrixStats')  # Row and Column wize statistics
library('mvtnorm')      # Multivariate normal distribution
library('pbapply')      # Print progress bar for the apply family 

# --- Set working directory ---
setwd("~/master-thesis/experiments/covariance")

# --- Load external functions ---
source('./R/utilityFunctions.R')
source('./R/covFunctions.R')

# --- Dataset names ---
problems <- c('nonsparse','sparse');

# --- Algorithm names ---
estimators <- c('emp','diag','oas','ss','ppca','spc','quic');

# --- Experiment parameters ---
n.runs <- 50;
N <- c(1000, 200, 100, 50, 10);

# --- Experiment ---
results <- lapply(problems, estimateCovariance, algs=estimators, 
                  samp.size=N, n=n.runs);
results <- do.call('rbind.data.frame', results);
rownames(results) <- NULL;

# --- Average results ---
results <- ddply(results, .(dataset, ratio, algorithm), summarise, 
                 l0.mean=mean(l0), l0.sd=sd(l0), l1.mean=mean(l1),
                 l1.sd=sd(l1), l2.mean=mean(l2), l2.sd=sd(l2), l3.mean=mean(l3),
                 l3.sd=sd(l3), eigdiff.mean=mean(eigdiff), 
                 eigdiff.sd=sd(eigdiff), eigcos.mean=mean(eigcos), 
                 eigcos.sd=sd(eigcos), time.mean=mean(time), 
                 time.sd=sd(time));
write.csv(x=results, file='results.csv', quote=FALSE, row.names=FALSE)