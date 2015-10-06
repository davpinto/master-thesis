# ============================================================
# rgbclass: Regularized Gaussian Bayes Classifiers
#
# AUTHOR:
# David Pinto
#
# CREATION DATE:
# June 27, 2015 at 11:30
# ============================================================

# --- Regularized Gaussian-Bayes classifier ---
# estim <- c('emp','diag','oas','ss','ppca','spc','quic')
rgb <- function(x, y, estim='oas') {
    # --- Check target class ---
    try(
        if(!is.factor(y)) {
            stop('y must be a factor array')
        }
    )

    # --- Covariance estimator ---
    covMat <- covEstimator(estim);

    # --- In-class Gaussian modeling ---
    gaussModel <- function(cls.label, x, y) {
        # -- Extract class examples --
        N  <- nrow(x);
        p  <- ncol(x);
        x  <- x[y==cls.label, , drop=FALSE];
        Nk <- nrow(x);

        # -- Class prior probability --
        # - ML estimator -
        pk.ml <- Nk/N;
        # - James-Stein Shrinkage estimator -
        # [Hausser - 2009] Entropy inference and the James–Stein estimator
        shrIntensity <- function(y) {
            N <- length(y);
            K <- length(levels(y));
            Nk <- as.numeric(table(y));
            shr <- ( 1-sum((Nk/N)^2) )/( (N-1)*sum((1/K-Nk/N)^2) );
            return( max(0,min(shr,1)) )
        }
        k <- length(levels(y));
        pk.tar <- 1/k;
        lambda <- shrIntensity(y);
        pk.shr <- lambda*pk.tar + (1-lambda)*pk.ml;

        # -- Covariance matrix --
        sig <- covMat(x);

        # -- Vector of means --
        # - MLE -
        mu.mle <- colMeans(x);
        # - Regularized MLE -
        # [DeMiguel - 2013] Size Matters: Optimal Calibration of shrinkage
        # estimators
        mu.tar <- rep(mean(mu.mle), p);
        sig2   <- mean(apply(x, 2, var)); # sum(diag(sig))/p;
        alpha  <- (p/Nk)*sig2/((p/Nk)*sig2 + norm(mu.tar-mu.mle, '2')^2);
        alpha  <- max(0,min(alpha,1));
        mu     <- (1-alpha)*mu.mle + alpha*mu.tar;

        return( list(k=cls.label, log.p=pk.shr, mu=mu, sig=sig) )
    }

    # --- Build Gaussian-Bayes learner ---
    model <- lapply(levels(y), FUN=gaussModel, x=x, y=y);

    # --- Normalize priors ---
    priors <- sapply(model, function(m) m$log.p, simplify=TRUE);
    model <- lapply(model, function(m, p) {
        # PS: Using logarithms to avoid underflow problems
        m$log.p <- log(m$log.p/sum(p));
        return(m)
    }, p=priors)

    # --- Return model ---
    model <- list(model=model, class.labels=levels(y), calib=NULL);
    return(structure(model, class='rgb'))
}

# --- Calibrate Two-Class RGB Probabilities ---
calib.rgb <- function(model, x, y) {
    # -- Model Prediction --
    pred.prob <- predict(model, x, type='prob')[,1];

    # -- Calibrate --
    calib.model <- CORElearn::calibrate(y, pred.prob, class1=1,
                                        method='isoReg', noBins=10,
                                        assumeProbabilities=TRUE);
    model$calib <- calib.model;

    return(model)
}

# --- Predict Regularized Gaussian-Bayes classifier ---
predict.rgb <- function(model, x, type='class') {
    # --- Compute posterior probabilities ---
    logPostProb <- function(model, x)
    {
        # --- Bayes Rule ---
        log.dens <- mvnfast::dmvn(x, model$mu, model$sig, log=TRUE, ncores=2)
        return(log.dens + model$log.p)
    }

    # --- Log trick to avoid under/overflow ---
    resp <- lapply(model$model, logPostProb, x=x);
    resp.log <- do.call('cbind', resp);
    resp <- sweep(resp.log, 1, matrixStats::rowMaxs(resp.log),
                  '-'); # set max log-posterior as zero

    # --- Assign labels ---
    label.idx <- apply(resp, 1, which.max);
    pred <- factor(model$class.labels[label.idx], levels=model$class.labels);

    # --- Probabilities ---
    post.prob <- exp(resp); # exponentiate log-posterior
    post.prob <- sweep(post.prob, 1, rowSums(post.prob), '/'); # normalize
    colnames(post.prob) <- model$class.labels;

    # --- Return ---
    switch(type,
        'class' = {
            return(pred)
        },
        'response' = {
            return(resp.log)
        },
        'prob' = {
            return(post.prob)
        },
        'both' = {
            return(list(pred=pred, prob=post.prob))
        }
    )
}

# --- Boosted Regularized Gaussian Bayes ---
# estim <- c('emp','diag','oas','ss','ppca','spc','quic')
# eta: 1   Freund and Schapire (1996)
# eta: 0.5 Breiman (1998)
brgb <- function(x, y, estim='oas', eta=0.5, iter=20, calib=FALSE) {
    # --- Check number of classes ---
    try(
        if(!is.factor(y)) {
            stop('y must be a factor array')
        } else if (length(levels(y))>2) {
            stop('y must be a dichotomous factor array')
        }
    )

    # --- Initialize weights ---
    N <- nrow(x);
    w <- rep(1/N, N);

    # --- Boosting iterations ---
    models <- vector('list', length=iter);
    alpha <- numeric(iter);
    e <- numeric(iter);
    alpha.oob <- numeric(iter);
    for(i in 1:iter) {
        # -- Bootstrap resampling --
        boot.idx <- sample(N, replace=TRUE, prob=w);
        x.w <- x[boot.idx, , drop=FALSE];
        y.w <- y[boot.idx];
        model <- rgb(x.w, y.w, estim=estim);
        # - "Out of the bag" instances -
        if(calib) {
            x.oob <- x[-unique(boot.idx), , drop=FALSE];
            y.oob <- y[-unique(boot.idx)];
            model <- calib.rgb(model, x.oob, y.oob);
        }

        # -- Model predictions --
        y.hat <- predict(model, x);

        # -- Misclassification probability --
        eps <- 1e-15; # probability threshold
        e[i] <- sum(w*as.numeric(y.hat != y));
        e[i] <- max(eps, min(e[i], 1-eps));

        # -- Model performance --
        alpha[i] <- eta*log((1-e[i])/e[i]);

        # -- Update weights --
        w <- w*exp(alpha[i]*ifelse(y==y.hat,-1,1));
        w <- w/sum(w);

        # -- Store model --
        models[[i]] <- model;
    }
    model <- list(boost=models, w=alpha, error=e, class.labels=levels(y));

    return( structure(model, class='brgb') )
}

# --- Predict Boosted Regularized Gaussian Bayes ---
predict.brgb <- function(model, x, type='class') {
    # -- Response matrix --
    resp <- numeric(nrow(x));

    # -- Average responses --
    for(i in seq_along(model$boost)) {
        # - Weighted Votes -
        # y.hat <- predict(model$boost[[i]], x);
        # y.bin <- as.matrix(table(1:length(y.hat), y.hat));
        # rownames(y.bin) <- NULL;
        # colnames(y.bin) <- NULL;
        # resp <- resp + y.bin*(model$w[i]);

        # - Weighted Average Probabilities -
        y.logp <- predict(model$boost[[i]], x, type='prob')[,1];
        # Calibrate probabilities 
        if(!is.null(model$boost[[i]]$calib)) {
            y.logp <- model$boost[[i]]$calib$calProb[
                findInterval(y.logp, c(0, model$boost[[i]]$calib$interval),
                             all.inside=TRUE)];
        }
        # Invert output of classifiers with misclassification prob. < 0.5
        if (model$w[i] >= 0) {
          resp <- resp + y.logp*(model$w[i]);
        } else {
          resp <- resp + (1-y.logp)*(-model$w[i]);
        }

    }
    resp <- resp/sum(abs(model$w));

    # --- Assign Labels ---
    label.idx <- apply(cbind(resp,1-resp), 1, which.max);
    pred <- factor(model$class.labels[label.idx], levels=model$class.labels);

    # --- Return ---
    switch(type,
       'class' = {
           return(pred)
       },
       'response' = {
           return(resp)
       }
    )
}

# --- Multiclass Boosted Regularized Gaussian Bayes ---
# [Rifkin - 2004] In Defense of One-Vs-All Classification
# estim <- c('emp','diag','oas','ss','ppca','spc','quic')
multi.brgb <- function(x, y, estim='oas', eta=0.5, iter=20, calib=TRUE) {
    # -- Boosting parameters --
    brgb.params <- list(estim=estim, eta=eta, iter=iter, calib=calib);

    # -- Binary boosted classifier --
    boost.binary <- function(label, x.tr, y.tr, params) {
        y.tr <- factor(as.numeric(y.tr==label), levels=c(1,0));
        model <- brgb(x.tr, y.tr, estim=params$estim, eta=params$eta,
                      iter=params$iter, calib=calib);
        return( list(model=model, class=label) )
    }
    if (exists('cl')) {
        models <- parallel::parLapply(cl, levels(y), boost.binary, x.tr=x, y.tr=y,
                                      params=brgb.params);
    } else {
        models <- lapply(levels(y), boost.binary, x.tr=x, y.tr=y, 
                         params=brgb.params);
    }
    model <- list(model=models, class.labels=levels(y));

    return( structure(model, class='multibrgb') )
}

# --- Predict Multiclass Boosted Regularized Gaussian Bayes ---
predict.multibrgb <- function(model, x, type='class') {
    # -- Binary responses --
    boost.binary.predict <- function(model, x.te) {
        # Continuous output
        y.hat <- predict(model$model, x.te, type='response');
        return(y.hat)
    }
    if (exists('cl')) {
        resp <- parallel::parLapply(cl, model$model, boost.binary.predict, 
                                    x.te=x);
    } else {
        resp <- lapply(model$model, boost.binary.predict, x.te=x);
    }
    resp <- do.call('cbind', resp);

    # --- Assign Labels ---
    label.idx <- apply(resp, 1, which.max);
    pred <- factor(model$class.labels[label.idx], levels=model$class.labels);

    # --- Probabilities ---
    post.prob <- sweep(resp, 1, rowSums(resp), '/'); # normalize
    colnames(post.prob) <- model$class.labels;

    # --- Return ---
    switch(type,
       'class' = {
           return(pred)
       },
       'response' = {
           return(resp)
       },
       'prob' = {
           return(post.prob)
       },
       'both' = {
           return(list(pred=pred, prob=post.prob))
       }
    )
}

# --- Improved covariance estimators ---
covEstimator <- function(type='Sample') {

    # --- Select estimator ---
    switch(type,
        'emp' = { # Empirical estimator
            function(x, w) {
                S <- empCov(x, w=w);
                return(S$cov)
            }
        },

        'diag' = { # Diagonal estimator with unequal variances
            function(x, w) {
                S <- diagCov(x, w=w);
                return(S$cov)
            }
        },

        'oas' = { # Oracle-Approximating shrinkage estimator
            function(x, w) {
                S <- oasCov(x, w=w);
                return(S$cov)
            }
        },

        'oas-diag' = { # Diagonal Oracle-Approximating shrinkage estimator
            function(x, w) {
                S <- oasCov(x, w=w);
                S <- diag(diag(S$cov), ncol=ncol(x), nrow=ncol(x));
                return(S)
            }
        },

        'ss' = { # Schafer-Strimmer shrinkage estimator
            function(x, w) {
                S <- ssCov(x, w=w);
                return(S$cov)
            }
        },

        'ppca' = { # Probabilistic PCA estimator
            function(x, w) {
                S <- ppcaCov(x);
                return(S$cov)
            }
        },

        'ppca-2' = { # Low-rank Probabilistic PCA estimator
            function(x, w) {
                S <- ppcaCov(x, k=2);
                return(S$cov)
            }
        },

        'spc' = { # Sparse PC's estimator
            function(x, w) {
                S <- spcCov(x, w);
                return(S$cov)
            }
        },
        
        'quic' = { # Quadratic Glasso estimator
            function(x, w) {
                S <- quicCov(x, w);
                return(S$cov)
            }
        }
    )
}

# --- Fast multivariate normal density ---
# Reference: http://gallery.rcpp.org/articles/dmvnorm_arma/
fastDMVN <- function(x, sig, mu, logd=TRUE) {
    p <- ncol(x);
    rooti <- backsolve(chol(sig), diag(p));
    quads <- colSums((crossprod(rooti,(t(x)-mu)))^2);
    log.dens <- -(p/2)*log(2*pi) + sum(log(diag(rooti))) - .5*quads;
    log.dens <- log.dens - max(log.dens);
    if(logd==FALSE) {
        return( exp(log.dens) )
    }
    return(log.dens)
}