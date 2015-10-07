# --- Dataset classification ---
classifyDataset <- function(dataset, algs, k, n) {
    # -- Load data --
    load(paste('./data/data_', dataset, '.RData', sep=''))
    cat(paste('Dataset: ', dataset, '...\n', sep=''))

    # -- Normalize data --
    x.means <- colMeans(x);
    x.sds <- apply(x, 2, sd);
    x <- sweep(x, 2, x.means, '-'); # center data
    x <- sweep(x, 2, x.sds, '/');   # scale data

    # -- Run classifiers --
    options("pbapply.pb"="txt")
    results <- pbapply::pblapply(rep(k, n), kFoldCV, x=x, y=y, algs=algs,
                                 dataset=dataset);

    # -- Merge results --
    results <- do.call('rbind.data.frame', results);

    return(results)
}

# --- k Fold Cross-Validation ---
kFoldCV <- function(k, x, y, algs, dataset) {
    # -- Split into k roughly equal subsets --
    # The class priors are maintained
    k.folds <- caret::createFolds(y, k=k, list=TRUE, returnTrain=FALSE);

    # -- Run CV --
    results <- vector('list', k);
    for(i in seq_along(results)) {
        # - Split in training and test sets --
        test.idx <- k.folds[[i]];
        x.test <- x[test.idx, , drop=FALSE];
        x.test <- x[test.idx, , drop=FALSE];
        y.test <- y[test.idx];
        x.train <- x[-test.idx, , drop=FALSE];
        y.train <- y[-test.idx];

        # - Remove constant variables -
        const.feat <- c(which(colSds(x.train)==0), which(colSds(x.test)==0));
        const.feat <- unique(const.feat);
        if(length(const.feat)>0) {
            x.train <- x.train[, -const.feat, drop=FALSE];
            x.test <- x.test[, -const.feat, drop=FALSE];
        }

        # - Train classifiers -
        models <- lapply(algs, trainClassifier, x=x.train, y=y.train);

        # - Test classifiers -
        results[[i]] <- lapply(models, testClassifier, x=x.test, y=y.test,
                               dataset=dataset);
        results[[i]] <- do.call('rbind.data.frame', results[[i]]);
    }
    results <- do.call('rbind.data.frame', results);

    return(results)
}

# --- Procedure to train classifiers ---
trainClassifier <- function(alg.name, x, y) {
    # -- Select and train classifier --
    t.elapsed <- system.time({
        switch(alg.name,
               'GaussNaiveBayes' = {
                   model <- e1071::naiveBayes(x=x, y=y, laplace=0);
               },

               'KernelNaiveBayes' = {
                   model <- klaR::NaiveBayes(x, y, usekernel=TRUE, fL=1);
               },

               'LinearDiscriminant' = {
                   options(warn=-1)
                   model <- sda::sda(x, y, diagonal=FALSE, verbose=FALSE);
                   options(warn=0)
               },

               'GlmNet' = {
                   model <- cv.glmnet(x=x, y=y, family='multinomial', nfolds=5,
                                      nlambda=15, type.measure="class",
                                      parallel=TRUE);
               },

               'Cart' = {
                   dt <- data.frame(x=x, y=y);
                   model <- rpart::rpart(y~., method="class",
                                         parms=list(split='gini'), data=dt,
                                         control=rpart.control(xval=5));
                   model <- rpart::prune(model, cp=model$cptable[which.min(
                       model$cptable[,"xerror"]),"CP"]);
               },

               'C4.5DecisionTree' = {
                   dt <- data.frame(x=x, y=y);
                   model <- RWeka::J48(y~., dt, control=Weka_control(U=FALSE,
                                                                     C=0.25));
               },

               'LinearSvm' = {
                   options(warn=-1)
                   c.path <- c(0.01, 0.1, 0.5, 1, 2, 5, 10, 15, 20);
                   cost <- e1071::tune.svm(x, y, type='C-classification',
                                           kernel='linear', cost=c.path,
                                           tunecontrol=tune.control(
                                               sampling='cross', cross=5)
                   )$best.param;
                   model <- e1071::svm(x, y, type='C-classification',
                                       kernel='linear', cost=cost, scale=TRUE,
                                       cross=0, probability=TRUE);
                   options(warn=0)
               },

               'RegGaussBayes' = {
                   model <- try({
                       rgb(x=x, y=y, estim='oas');
                   }, silent=TRUE);
               },

               'RegGaussBayes-diag' = {
                   model <- try({
                       rgb(x=x, y=y, estim='oas-diag');
                   }, silent=TRUE);
               },

               'RegGaussBayes-lowrank' = {
                   model <- try({
                       rgb(x=x, y=y, estim='ppca-2');
                   }, silent=TRUE);
               },

               'BoostedRgb' = {
                   model <- try({
                       multi.brgb(x, y, estim='oas', eta=0.5, iter=20,
                                  calib=FALSE);
                   }, silent=TRUE);
               },

               'BoostedRgb-diag' = {
                   model <- try({
                       multi.brgb(x, y, estim='oas-diag', eta=0.5, iter=20,
                                  calib=FALSE);
                   }, silent=TRUE);
               },

               'BoostedRgb-lowrank' = {
                   model <- try({
                       multi.brgb(x, y, estim='ppca-2', eta=0.5, iter=20,
                                  calib=FALSE);
                   }, silent=TRUE);
               },

               'BoostedRgb-iso' = {
                   model <- try({
                       multi.brgb(x, y, estim='oas', eta=0.5, iter=20,
                                  calib=TRUE);
                   }, silent=TRUE);
               },

               'GaussProcess' = {
                   options(warn=-1)
                   sig <- median( sapply(1:100, function(i, x, sc)
                       kernlab::sigest(x=x, scaled=sc)[2], x=x, sc=FALSE,
                       simplify='array') );
                   model <- kernlab::gausspr(x, y, scaled=FALSE, kernel="rbfdot",
                                             kpar=list(sigma=sig), var=1, tol=1e-4,
                                             type='classification');
                   options(warn=0)
               },

               'NonlinearRbfSvm' = {
                   options(warn=-1)
                   sig <- median( sapply(1:100, function(i, x, sc)
                       kernlab::sigest(x=x, scaled=sc)[2], x=x, sc=FALSE,
                       simplify='array') );
                   c.path <- c(0.01, 0.1, 0.5, 1, 2, 5, 10, 15, 20);
                   cost <- e1071::tune.svm(x, y, type='C-classification',
                                           kernel='radial', cost=c.path, gamma=sig,
                                           tunecontrol=tune.control(
                                               sampling='cross', cross=5)
                   )$best.param[2];
                   model <- e1071::svm(x, y, type='C-classification',
                                       kernel='radial', cost=cost, scale=TRUE,
                                       cross=0, gamma=sig, probability=TRUE);
                   options(warn=0)
               },

               'ExtraTrees' = {
                   model <- extraTrees::extraTrees(x, y, ntree=150, numThreads=4);
               },

               'Adaboost' = {
                   dt <- data.frame(x=x, y=y);
                   model <- RWeka::AdaBoostM1(y~., dt, control=Weka_control(
                       W="J48", I=50));
               },

               'SvmSofia' = {
                   model <- svmsofia(x=x, y=y, ncores=4);
               },

               'GradientBoosting' = {
                   options(warn=-1)
                   model <- gbm::gbm.fit(x=x, y=y, keep.data=FALSE,
                                         distribution='multinomial', n.trees=150,
                                         shrinkage=5e-2, bag.fraction=0.5,
                                         interaction.depth=5, n.minobsinnode=10,
                                         verbose=FALSE);
                   options(warn=0)
               },

               'RandomForest' = {
                   model <- randomForest::randomForest(x=x, y=y, ntree=150,
                                                       importance=FALSE);
               }
        )
    });

    return( list(name=alg.name, model=model, time=t.elapsed[3]) )
}

# --- Procedure to test classifiers ---
testClassifier <- function(model, x, y, dataset) {
    # -- Extract model information --
    alg.name <- model$name;
    train.time <- model$time;
    model <- model$model;

    # -- Select prediction model --
    switch(alg.name,
           'GaussNaiveBayes' = {
               y.hat  <- predict(model, x);
               y.prob <- predict(model, x, type='raw');
           },

           'KernelNaiveBayes' = {
               options(warn=-1)
               resp <- predict(model, x, threshold=0);
               options(warn=0)
               y.hat <- resp$class;
               y.prob <- resp$posterior;
           },

           'LinearDiscriminant' = {
               resp <- predict(model, x, verbose=FALSE);
               y.hat <- resp$class;
               y.prob <- resp$posterior;
           },

           'GlmNet' = {
               y.hat <- factor( predict(model, x, type='class', s='lambda.min')[,1],
                                levels=levels(y) );
               y.prob <- predict(model, x, type='response', s='lambda.min')[,,1];
           },

           'Cart' = {
               dt <- data.frame(x=x);
               y.prob <- predict(model, dt);
               y.hat <- factor( apply(y.prob, 1, function(row, labels)
                   labels[which.max(row)], labels=levels(y)), levels=levels(y) );
           },

           'C4.5DecisionTree' = {
               dt <- data.frame(x=x, y=rep(0,nrow(x)));
               y.hat <- predict(model, dt, type='class');
               y.prob <- predict(model, dt, type='probability');
           },

           'LinearSvm' = {
               y.hat <- predict(model, x, probability=FALSE);
               y.prob <- attr(predict(model, x, probability=TRUE),
                              "probabilities");
               y.prob <- y.prob[, levels(y)]; # ensure labels order
           },

           'RegGaussBayes' = {
               if(class(model) != 'try-error') {
                   resp <- predict(model, x, type='both');
                   y.hat <- resp$pred;
                   y.prob <- resp$prob;
               } else {
                   return( data.frame(dataset=dataset, algorithm=alg.name,
                                      acc=NA, auc=NA, loss=NA, time=NA) )
               }
           },

           'RegGaussBayes-diag' = {
               if(class(model) != 'try-error') {
                   resp <- predict(model, x, type='both');
                   y.hat <- resp$pred;
                   y.prob <- resp$prob;
               } else {
                   return( data.frame(dataset=dataset, algorithm=alg.name,
                                      acc=NA, auc=NA, loss=NA, time=NA) )
               }
           },

           'RegGaussBayes-lowrank' = {
               if(class(model) != 'try-error') {
                   resp <- predict(model, x, type='both');
                   y.hat <- resp$pred;
                   y.prob <- resp$prob;
               } else {
                   return( data.frame(dataset=dataset, algorithm=alg.name,
                                      acc=NA, auc=NA, loss=NA, time=NA) )
               }
           },

           'BoostedRgb' = {
               if(class(model) != 'try-error') {
                   resp <- predict(model, x, type='both');
                   y.hat <- resp$pred;
                   y.prob <- resp$prob;
               } else {
                   return( data.frame(dataset=dataset, algorithm=alg.name,
                                      acc=NA, auc=NA, loss=NA, time=NA) )
               }
           },

           'BoostedRgb-diag' = {
               if(class(model) != 'try-error') {
                   resp <- predict(model, x, type='both');
                   y.hat <- resp$pred;
                   y.prob <- resp$prob;
               } else {
                   return( data.frame(dataset=dataset, algorithm=alg.name,
                                      acc=NA, auc=NA, loss=NA, time=NA) )
               }
           },

           'BoostedRgb-lowrank' = {
               if(class(model) != 'try-error') {
                   resp <- predict(model, x, type='both');
                   y.hat <- resp$pred;
                   y.prob <- resp$prob;
               } else {
                   return( data.frame(dataset=dataset, algorithm=alg.name,
                                      acc=NA, auc=NA, loss=NA, time=NA) )
               }
           },

           'BoostedRgb-iso' = {
               if(class(model) != 'try-error') {
                   resp <- predict(model, x, type='both');
                   y.hat <- resp$pred;
                   y.prob <- resp$prob;
               } else {
                   return( data.frame(dataset=dataset, algorithm=alg.name,
                                      acc=NA, auc=NA, loss=NA, time=NA) )
               }
           },

           'GaussProcess' = {
               y.hat <- predict(model, x);
               y.prob <- predict(model, x, type='probabilities');
           },

           'NonlinearRbfSvm' = {
               y.hat <- predict(model, x, probability=FALSE);
               y.prob <- attr(predict(model, x, probability=TRUE),
                              "probabilities");
               y.prob <- y.prob[, levels(y)]; # ensure labels order
           },

           'ExtraTrees' = {
               y.hat <- predict(model, x);
               y.prob <- predict(model, x, probability=TRUE);
           },

           'Adaboost' = {
               dt <- data.frame(x=x, y=rep(0,nrow(x)));
               y.hat <- predict(model, dt, type='class');
               y.prob <- predict(model, dt, type='probability');
           },

           'SvmSofia' = {
               resp <- predict(model, x);
               y.hat <- resp$pred;
               y.prob <- resp$prob;
           },

           'GradientBoosting' = {
               y.prob <- predict(model, newdata=x, n.trees=model$n.trees,
                                 type="response")[,,1];
               y.hat <- factor( apply(y.prob, 1, function(row, labels)
                   labels[which.max(row)], labels=levels(y)), levels=levels(y) );
           },

           'RandomForest' = {
               y.hat <- predict(model, x, 'response');
               y.prob <- predict(model, x, 'prob');
           },
    )

    # -- Evaluate model --
    model.perf <- classPerform(y, y.hat, y.prob);

    return( data.frame(dataset=dataset, algorithm=alg.name,
                       acc=model.perf$acc, auc=model.perf$auc,
                       loss=model.perf$loss, time=train.time) )
}

# --- Measure classification performance ---
classPerform <- function(y, pred=NULL, prob=NULL) {
    try(
        if(is.null(pred) || is.null(prob)) {
            stop('Inform both predicted labels and predicted probabilities')
        }
    )

    # --- Overall accuracy ---
    acc <- sum(pred==y)/length(y);

    # --- Overall area under the ROC curve (AUC) ---
    auc <- mean(rminer::mmetric(y=y, x=prob, metric='AUCCLASS'));

    # --- Log-loss ---
    log.loss <- multiLogLoss(y, prob);

    return(list(acc=acc, auc=auc, loss=log.loss))
}

# --- Multiclass log-loss ---
# Reference: https://www.kaggle.com/wiki/LogarithmicLoss
multiLogLoss <- function(y, prob) {
    # --- Probabilities threshold ---
    eps  <- 1e-15;
    prob <- pmin(pmax(prob, eps), 1-eps);

    # --- Label to binary matrix ---
    y <- lapply(y, function(current, all) as.numeric(all==current),
                all=levels(y));
    y <- do.call('rbind', y);

    # --- Compute log-loss ---
    log.loss <- (-1/nrow(y))*sum(y*log(prob));

    return(log.loss)
}

# --- Adaboost: one-against-all ---
# Fit model
adaboost <- function(x, y, ncores=1) {
    ada.binomial.fit <- function(label, x, y) {
        y <- as.numeric(y==label);
        model <- ada::ada(x, y, loss="logistic", type="gentle", iter=50,
                          nu=0.1);
        return( list(model=model, class=label) )
    }
    model <- parallel::mclapply(levels(y), ada.binomial.fit, x=x, y=y,
                                mc.cores=ncores);
    model <- list(ada=model, class.labels=levels(y));

    return( structure(model, class='adaboost') )
}
# Assign labels
predict.adaboost <- function(model, x) {
    ada.binomial.predict <- function(model, x) {
        y.hat <- predict(model$model, x, type='F'); # continuous output
        y.hat <- 1/(1 + exp(-y.hat)); # analogy with logistic regression
        return(y.hat)
    }
    y.pred <- lapply(model$ada, ada.binomial.predict, x=as.data.frame(x));
    y.pred <- do.call('cbind', y.pred);

    # --- Labels ---
    label.idx <- apply(y.pred, 1, which.max);
    y.hat <- factor(model$class.labels[label.idx], levels=model$class.labels);

    # --- Probabilities ---
    y.prob <- sweep(y.pred, 1, rowSums(y.pred), '/');
    colnames(y.prob) <- model$class.labels;

    return( list(pred=y.hat, prob=y.prob) )
}

# --- Sofia Fast Linear SVM: one-against-all ---
# Fit model
svmsofia <- function(x, y, ncores=1) {
    # -- Center and scale features --
    x.means <- colMeans(x);
    x.sds   <- apply(x, 2, sd);
    x <- sweep(x, 2, x.means, '-');
    x <- sweep(x, 2, x.sds, '/');
    norm.params <- list(center=x.means, scale=x.sds);

    # -- Multiclass to binary --
    sofia.binomial.fit <- function(label, x, y) {
        y <- as.numeric(y==label);
        y[y==0] <- -1;
        dt <- data.frame(x=x, y=y);
        model <- RSofia::sofia(y~., data=dt, learner_type="logreg-pegasos",
                               perceptron_margin_size=2,
                               loop_type='combined-roc');
        return( list(model=model, class=label) )
    }
    model <- lapply(levels(y), sofia.binomial.fit, x=x, y=y);
    # model <- parallel::mclapply(levels(y), sofia.binomial.fit, x=x, y=y,
                                # mc.cores=ncores);
    model <- list(svm=model, class.labels=levels(y), norm=norm.params);

    return( structure(model, class='svmsofia') )
}
# Assign labels
predict.svmsofia <- function(model, x) {
    # -- Center and scale test instances --
    x <- sweep(x, 2, model$norm$center, '-');
    x <- sweep(x, 2, model$norm$scale, '/');

    # -- Class evidence --
    sofia.binomial.predict <- function(model, x) {
        dt <- data.frame(x=x, y=rep(0, nrow(x)));
        y.hat <- predict(model$model, newdata=dt, prediction_type="logistic");
        return(y.hat)
    }
    y.pred <- lapply(model$svm, sofia.binomial.predict, x=x);
    y.pred <- do.call('cbind', y.pred);

    # -- Labels --
    label.idx <- apply(y.pred, 1, which.max);
    y.hat <- factor(model$class.labels[label.idx], levels=model$class.labels);

    # -- Probabilities --
    y.prob <- sweep(y.pred, 1, rowSums(y.pred), '/');
    colnames(y.prob) <- model$class.labels;

    return( list(pred=y.hat, prob=y.prob) )
}
