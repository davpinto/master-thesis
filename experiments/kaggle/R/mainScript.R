# --- Clear workspace ---
rm(list=ls())

# --- Set working directory ---
setwd("~/master-thesis/experiments/kaggle")

# --- Constants ---
nCores <- 4;

# --- Load packages ---
library('plyr')         # Data wrangling/munging
library('xgboost')      # xgboost library for gradient boosting
library('h2o')          # h2o library for fast and scalable ML algorithms
library('caret')        # K-fold Cross-Validation routine
library('CORElearn')    # Probability Calibration
library('rminer')       # Performance metrics
library('matrixStats')  # Row and Column wize statistics
library('mvnfast')      # Fast multivariate Normal density
library('pbapply')      # Print progress bar for the apply family
library('parallel')     # Paralell alternative to the apply family
cl <- parallel::makeCluster(nCores, type="SOCK");

# --- Load external functions ---
source('../classification/R/utilityFunctions.R')
source('../classification/R/rgbFunctions.R')
source('../covariance/R/covFunctions.R')
parallel::clusterExport(cl, ls())

# --- Create H2O Cluster ---
localH2O <- h2o.init(max_mem_size='2g', nthreads=nCores);

# --- Load data ---
data.set <- 'higgs'; # choose: 'higgs' (two-class) or 'otto' (multi-class)
load(paste('./data/data_', data.set, '.RData', sep=''))
x <- matrix(as.numeric(x), ncol=ncol(x));

# --- XGBoost Parameters ---
if(data.set == 'higgs') {
    xg.params <- list("objective" = "binary:logistic", "bst:eta" = 0.1,
                      "bst:max_depth" = 6, "eval_metric" = "auc", "silent" = 1,
                      "nthread" = nCores);
    h2o.family <- 'bernoulli';
} else { # dataset 'otto'
    xg.params <- list("objective" = "multi:softprob",
                      "eval_metric" = "mlogloss", "silent" = 1,
                      "num_class" = nlevels(y), "nthread" = nCores);
    h2o.family <- 'multinomial';
}

# --- Run experiment ---
n.folds <- 3; # Roughly 66% for training and 34% for testing
n.runs  <- 5; # Give a sample of 15 results
results.all <- vector('list', n.runs);
for (idx in 1:n.runs) {
    # -- Split Data --
    test.idx <- caret::createFolds(y, n.folds);

    # -- k-Fold CV --
    algs    <- c('RGB', 'BRGB', 'ISO-BRGB', 'XGBOOST', 'H2O-GNB',
                 'H2O-GBM', 'H2O-RF');
    nalgs   <- length(algs);
    results <- vector('list', n.folds);
    for (i in 1:n.folds) {
        str.status <- paste('\nTrial: ', idx, '; Fold: ', i, '\n', sep='');
        cat(str.status)

        # - Split data -
        x.te <- x[test.idx[[i]], , drop=FALSE];
        y.te <- y[test.idx[[i]]];
        x.tr <- x[-test.idx[[i]], , drop=FALSE];
        y.tr <- y[-test.idx[[i]]];

        # - H2O Data -
        train_hex <- as.h2o(data.frame(x=x.tr, y=y.tr), conn=localH2O);
        test_hex  <- as.h2o(data.frame(x=x.te, y=y.te), conn=localH2O);

        # - Regularized Gaussian Bayes -
        # 1. RGB (Regularized Gaussian Bayes)
        rgb.time <- system.time({
            rgb.model <- rgb(x.tr, y.tr, estim='oas');
        })[3];
        resp <- predict(rgb.model, x.te, type='both');
        rgb.perf <- classPerform(y.te, resp$pred, resp$prob);
        # 2. BRGB (Boosted Regularized Gaussian Bayes)
        brgb.time <- system.time({
            brgb.model <- multi.brgb(x.tr, y.tr, iter=20, eta=0.1, estim='oas',
                                     calib=FALSE);
        })[3];
        resp <- predict(brgb.model, x.te, type='both');
        brgb.perf <- classPerform(y.te, resp$pred, resp$prob);
        # 3. ISO-BRGB (BRGB calibrated with Isotonic Regression)
        iso.brgb.time <- system.time({
            iso.brgb.model <- multi.brgb(x.tr, y.tr, iter=20, eta=0.1,
                                         estim='oas', calib=TRUE);
        })[3];
        resp <- predict(iso.brgb.model, x.te, type='both');
        iso.brgb.perf <- classPerform(y.te, resp$pred, resp$prob);

        # - Extreme Boosting -
        xg.time <- system.time({
            xg.model <- xgboost::xgboost(booster='gbtree', param=xg.params,
                                         data=x.tr, label=as.integer(y.tr)-1,
                                         nrounds=100, verbose=FALSE);
        })[3];
        k <- nlevels(y.tr);
        y.prob <- predict(xg.model, x.te);
        if (k==2) {
            y.prob <- cbind(1-y.prob, y.prob);
        } else {
            y.prob <- matrix(y.prob, ncol=k, byrow=TRUE);
        }
        y.hat <- factor( apply(y.prob, 1, function(row, labels)
            labels[which.max(row)], labels=levels(y.tr)), levels=levels(y.tr) );
        xg.perf <- classPerform(y.te, y.hat, y.prob);

        # - H2O Algorithms -
        # 1. GNB (Gaussian Naive Bayes)
        gnb.time <- system.time({
            gnb.model <- h2o.naiveBayes(x=1:(ncol(train_hex)-1),
                                        y=ncol(train_hex),
                                        training_frame=train_hex, laplace=1,
                                        compute_metrics = FALSE);
        })[3];
        gnb.resp <- h2o.predict(gnb.model, newdata=test_hex);
        gnb.resp <- as.data.frame(gnb.resp);
        y.hat    <- gnb.resp$predict;
        y.prob   <- data.matrix(gnb.resp[,2:ncol(gnb.resp)]);
        gnb.perf <- classPerform(y.te, y.hat, y.prob);
        # 2. GBM (Gradient Boosting Machine)
        gbm.time <- system.time({
            gbm.model <- h2o.gbm(x=1:(ncol(train_hex)-1), y=ncol(train_hex),
                                 training_frame=train_hex,
                                 distribution=h2o.family, ntrees=100,
                                 max_depth=6, min_rows=10, learn_rate=0.1);
        })[3];
        gbm.resp <- h2o.predict(gbm.model, newdata=test_hex);
        gbm.resp <- as.data.frame(gbm.resp);
        y.hat    <- gbm.resp$predict;
        y.prob   <- data.matrix(gbm.resp[,2:ncol(gbm.resp)]);
        gbm.perf <- classPerform(y.te, y.hat, y.prob);
        # 3. RF (Random Forest)
        rf.time <- system.time({
            rf.model <- h2o.randomForest(x=1:ncol(x.tr), y=ncol(train_hex),
                                         training_frame=train_hex,
                                         classification=TRUE, ntrees=150,
                                         depth=10, mtries=-1,
                                         balance_classes=FALSE,
                                         importance=FALSE);
        })[3];
        rf.resp <- h2o.predict(rf.model, newdata=test_hex);
        rf.resp <- as.data.frame(rf.resp);
        y.hat    <- rf.resp$predict;
        y.prob   <- data.matrix(rf.resp[,2:ncol(rf.resp)]);
        rf.perf <- classPerform(y.te, y.hat, y.prob);

        # - Collect results -
        results[[i]] <- data.frame(dataset=rep(data.set, nalgs), algorithm=algs,
                                   acc=c(rgb.perf$acc, brgb.perf$acc,
                                         iso.brgb.perf$acc, xg.perf$acc,
                                         gnb.perf$acc, gbm.perf$acc,
                                         rf.perf$acc),
                                   auc=c(rgb.perf$auc, brgb.perf$auc,
                                         iso.brgb.perf$auc, xg.perf$auc,
                                         gnb.perf$auc, gbm.perf$auc,
                                         rf.perf$auc),
                                   loss=c(rgb.perf$loss, brgb.perf$loss,
                                          iso.brgb.perf$loss, xg.perf$loss,
                                          gnb.perf$loss, gbm.perf$loss,
                                          rf.perf$loss),
                                   time=c(rgb.time, brgb.time, iso.brgb.time,
                                          xg.time, gnb.time, gbm.time, rf.time));
    }

    results.all[[idx]] <- do.call('rbind.data.frame', results);
}
stopCluster(cl)
h2o.shutdown(conn=localH2O, prompt = FALSE)
results <- do.call('rbind.data.frame', results.all);
rownames(results) <- NULL;
results <- transform(results, algorithm=factor(algorithm, levels=algs));

# --- Average results ---
results <- results[complete.cases(results),];
results <- ddply(results, .(dataset, algorithm), summarise,
                 acc.mean=mean(acc), acc.sd=sd(acc), auc.mean=mean(auc),
                 auc.sd=sd(auc), loss.mean=mean(loss), loss.sd=sd(loss),
                 time.mean=mean(time), time.sd=sd(time));
write.csv(x=results, file='results.csv', quote=FALSE, row.names=FALSE)
