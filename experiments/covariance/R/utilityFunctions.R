# --- Covariance Estimation ---
estimateCovariance <- function(dataset, algs, samp.size, n) {
    # -- Load data --
    load(paste('./data/', dataset, '.RData', sep=''))
    cat(paste('Dataset: ', dataset, '...\n', sep=''))
    
    # --- Mean vector ---
    mu <- rep(0, ncol(Sig));
    
    # -- Run estimators --
    options("pbapply.pb"="txt")
    results <- pbapply::pblapply(rep(samp.size, each=n), oneRound, mu=mu,
                                 sig=Sig, algs=algs, dataset=dataset);
    
    # -- Save results --
    results <- do.call('rbind.data.frame', results);
    save(results, file=paste('./datasets/results/result_', dataset, '.RData', 
                             sep=''))
    
    return(results)
}

# --- Experiment iteration ---
oneRound <- function(N, mu, sig, algs, dataset) {
    # -- Sample data --
    x <- mvtnorm::rmvnorm(N, mu, sig);
    p <- ncol(x);
    
    # -- Run Estimators --
    S <- lapply(algs, runEstimator, x=x);
    
    # -- Evaluate Estimators --
    results <- lapply(S, evalEstimator, ratio=p/N, sig=sig, dataset=dataset);
    results <- do.call('rbind.data.frame', results);
    
    return(results)
}

# --- Procedure to run the estimators ---
runEstimator <- function(alg.name, x) {
    # -- Select and run estimator --
    t.elapsed <- system.time({
        switch(alg.name,
           'emp' = { # Sample/empirical estimator
               S <- cov(x);
               # [Higham - 1988] Computing a nearest symmetric positive semidefinite matrix
               if(ncol(x)>=nrow(x)) {
                   S <- corpcor::make.positive.definite(S);
               }
           },
           
           'diag' = { # Diagonal with unequal variances estimator
               S <- diag(matrixStats::colVars(x));
           },
           
           'ss' = { # Schafer-Strimmer estimator
               S <- ssCov(x)$cov;
           },
           
           'oas' = { # Oracle-approximating shrinkage estimator 
               S <- oasCov(x)$cov;
           },
           
           'ppca' = { # Probabilistic PCA
               S <- ppcaCov(x)$cov;
           },
           
           'spc' = { # Sparse Probabilistic PCA
               S <- spcCov(x)$cov;
           },
           
           'quic' = { # Quadratic Graphical Lasso
               S <- quicCov(x)$cov;
           }
        )
    });
    
    return( list(name=alg.name, estim=S, time=t.elapsed[3]) )
}

# --- Procedure to evaluate estimators ---
evalEstimator <- function(model, ratio, sig, dataset) {
    # -- Extract model information --
    alg.name <- model$name;
    run.time <- model$time;
    estim <- model$estim;
    
    # -- Entropy loss --
    p <- dim(sig)[1];
    cov.isig <- estim %*% chol2inv(chol(sig));
    L1 <- sum(diag(cov.isig)) - log(det(cov.isig)) - p;
    
    # -- Quadratic loss --
    L2 <- sum( diag(cov.isig - diag(1, nrow=p, ncol=p)) )^2;
    
    # -- RMSE --
    rmse <- norm(estim-sig, 'F')/p;
    rmse <- rmse/diff(range(sig)); # Normalized RMSE
    
    # -- Sparsity --
    L0 <- sum(estim==0)/(p^2);
    
    # -- Eigenstructure --
    S.eig <- eigen(estim, symmetric=TRUE);
    Sig.eig <- eigen(sig, symmetric=TRUE);
    
    # -- Absolute difference between maximum eigenvalues --
    eigdiff <- abs( max(S.eig$values)-max(Sig.eig$values) );
    
    # -- Absolute cosine between first eigenvector ---
    v1 <- S.eig$vectors[,1];
    v2 <- Sig.eig$vectors[,1];
    eigcos <- abs( sum(v1*v2) / ( sqrt(sum(v1 * v1)) * sqrt(sum(v2 * v2)) ) );
    
    return( data.frame(dataset=dataset, ratio=ratio, algorithm=alg.name, l0=L0, 
                       l1=L1, l2=L2, l3=rmse, eigdiff=eigdiff, eigcos=eigcos,
                       time=run.time) )
}