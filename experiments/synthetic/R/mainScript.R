# --- Clear workspace ---
rm(list=ls())

# --- Set working directory ---
setwd("~/master-thesis/experiments/synthetic")

# --- Load packages ---
library('plyr')         # Data wrangling/munging
library('glasso')       # Graphical Lasso algorithm
library('huge')         # Graphical Lasso selection
library('QUIC')         # Quadratic Graphical Lasso algorithm
library('caret')        # K-fold Cross-Validation routine
library('Metrics')      # Performance metrics
library('matrixStats')  # Row and Column wize statistics
library('mvnfast')      # Fast multivariate Normal density
library('pbapply')      # Print progress bar for the apply family 
library('ggplot2')

# --- Load external functions ---
source('../covariance/R/covFunctions.R')
source('../classification/R/rgbFunctions.R')
source('../classification/R/utilityFunctions.R')

# --- Dataset ---
problems <- c('twonorm','threenorm','ringnorm','waveform','waveformnoise',
              'corelearn','xor');

# --- Experiment parameters ---
tr.size <- 300;
n.runs  <- 30;

# --- Classification experiment ---
classHoldOutCV <- function(rep.idx, x, y, ratio, data) {
    # --- Split Data ---
    train.idx <- caret::createDataPartition(y, p=ratio, list=FALSE);
    x.tr <- x[train.idx, ];
    x.te <- x[-train.idx, ];
    y.tr <- y[train.idx];
    y.te <- y[-train.idx];
    
    # --- Hold-out CV ---
    algs <- c('diag', 'emp', 'oas', 'ss', 'ppca', 'spc', 'quic');
    trainTestRgb <- function(alg, x1, y1, x2, y2, data) {
        train.time <- system.time({
            model <- rgb(x1, y1, estim=alg);
        })
        resp <- predict(model, x2, type='both');
        y.hat <- resp$pred;
        y.prob <- resp$prob;
        model.perf <- classPerform(y2, y.hat, y.prob);
        return( data.frame(dataset=data, algorithm=alg, 
                           acc=model.perf$acc, auc=model.perf$auc, 
                           loss=model.perf$loss, time=train.time[3]) )
    }
    results <- lapply(algs, trainTestRgb, x1=x.tr, y1=y.tr, x2=x.te, y2=y.te,
                      data=data);
    results <- do.call('rbind.data.frame', results);
    rownames(results) <- NULL;
    
    return(results)
}
# -- Run hold-out CV for each dataset --
results <- lapply(problems, function(problem) {
    
    cat(paste('\nProblem: ', problem, '...\n', sep=''))
    
    # - Load data -
    load(paste('./data/data_', problem, '.RData', sep=''))
    x <- scale(x);
    N <- nrow(x);
    
    # - Run hold-out CV -
    results <- pbapply::pblapply(1:n.runs, classHoldOutCV, x, y, tr.size/N,
                                 problem);
    results <- do.call('rbind.data.frame', results);
    rownames(results) <- NULL;
    
    return(results)
});
results <- do.call('rbind.data.frame', results);
rownames(results) <- NULL;

# --- Format results ---
results <- transform(results, dataset=as.factor(dataset), 
                     algorithm=factor(algorithm, levels=c('diag','emp','oas',
                                                          'ss','ppca','spc',
                                                          'quic')));

# --- Average results ---
results <- results[complete.cases(results),];
results <- ddply(results, .(dataset, algorithm), summarise, 
                 acc.mean=mean(acc), acc.sd=sd(acc), auc.mean=mean(auc),
                 auc.sd=sd(auc), loss.mean=mean(loss), loss.sd=sd(loss),
                 time.mean=mean(time), time.sd=sd(time));
write.csv(x=results, file='results.csv', quote=FALSE, row.names=FALSE)