# --- Clear workspace ---
rm(list=ls())

# --- Set working directory ---
setwd("~/master-thesis/experiments/classification")

# --- Set number of cores ---
nCores <- 4;

# --- Load packages ---
library('plyr')         # Data wrangling/munging
library('randomForest') # Random Forest algorithm
library('extraTrees')   # Extra Trees algorithm
library('gbm')          # Gradient Boosting Machine algorithm
library('glmnet')       # Regularized GLMs
library('e1071')        # Gaussian Naive Bayes and SVM algorithms
library('klaR')         # Kernel Naive Bayes algorithm
library('rpart')        # Cart Decision Tree
library('ada')          # Discrete, Real and Gentle LogitBoost algorithms
library('RWeka')        # Adaboost.M1 and C4.5 Decision Tree algorithms
library('kernlab')      # Heuristic to select the radial kernel parameter
library('RSofia')       # Fast Linear SVM
library('QUIC')         # Quadratic Graphical Lasso algorithm
library('sda')          # Shrinkage Linear Discriminant Analysis
library('caret')        # K-fold Cross-Validation routine
library('CORElearn')    # Probability Calibration
library('rminer')       # Performance metrics
library('matrixStats')  # Row and Column wize statistics
library('mvnfast')      # Fast multivariate Normal density
library('pbapply')      # Print progress bar for the apply family
library('doMC')         # Paralell alternative to the apply family
registerDoMC(cores=nCores)
library('parallel')     # Paralell alternative to the apply family
cl <- parallel::makeCluster(nCores, type="SOCK");

# --- Load external functions ---
source('./R/utilityFunctions.R')
source('./R/rgbFunctions.R')
source('../covariance/R/covFunctions.R')
parallel::clusterExport(cl, ls())

# --- Dataset names ---
# All categorical variables have been transformed into dummy variables
# All NAs(missing values) have been removed
problems <- sapply(list.files('./data/', pattern='data_'), gsub,
                   pattern='.RData', replacement='', simplify='array');
problems <- sapply(problems, gsub, pattern='data_', replacement='',
                   simplify='array');
problems <- c('sonar') # comment this line to run for all datasets

# --- Algorithm names ---
classifiers <- c('GaussNaiveBayes','KernelNaiveBayes','LinearDiscriminant',
                 'GlmNet','Cart','LinearSvm','RegGaussBayes-diag',
                 'RegGaussBayes-lowrank','RegGaussBayes','BoostedRgb-diag',
                 'RegGaussBayes-lowrank','RegGaussBayes','GaussProcess',
                 'NonlinearRbfSvm','ExtraTrees','Adaboost','GradientBoosting',
                 'RandomForest'); # remove names to left out some algorithms

# --- Experiment parameters ---
k.folds <- 5; # Split data into subsets with roughly 20% of the instances
n.runs  <- 6; # To get a sample of 30 results

# --- Classify problems ---
results <- lapply(problems, classifyDataset, algs=classifiers, k=k.folds,
                  n=n.runs);
results <- do.call('rbind.data.frame', results);
rownames(results) <- NULL;
parallel::stopCluster(cl)

# --- Average results ---
results <- results[complete.cases(results),];
results <- ddply(results, .(dataset, algorithm), summarise,
                 acc.mean=mean(acc), acc.sd=sd(acc), auc.mean=mean(auc),
                 auc.sd=sd(auc), loss.mean=mean(loss), loss.sd=sd(loss),
                 time.mean=mean(time), time.sd=sd(time));
write.csv(x=results, file='results.csv', quote=FALSE, row.names=FALSE)
