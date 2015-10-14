# --- Empirical or Sample Estimator ---
empCov <- function(x, w) {
    if(!missing(w)) {
        Sig <- cov.wt(x, wt=w)$cov;
    } else {
        Sig <- cov(x);
    }
    Sig <- corpcor::make.positive.definite(Sig);
    
    return( list(cov=Sig) )
}

# --- Diagonal with Unequal variances Estimator ---
diagCov <- function(x, w) {
    if(!missing(w)) {
        S <- cov.wt(x, wt=w)$cov;
    } else {
        S <- cov(x);
    }
    S <- corpcor::make.positive.definite(S);
    Sig <- diag(diag(S), ncol=ncol(x), nrow=ncol(x));
    
    return( list(cov=Sig) )
}

# --- Ledoit-Wolf Shrinkage Estimator ---
lwCov <- function(x, alpha, w) {
    # --- Data dimension ---
    N <- nrow(x);
    p <- ncol(x);
    
    # --- Sample weights ---
    if(missing(w)) {
        w <- rep(1/N, N);
    } else {
        w <- w/sum(w);
    }
    
    # --- Sample estimator ---
    S <- cov.wt(x, wt=w)$cov;
    
    # --- Target estimator ---
    T <- diag(mean(diag(S)), ncol=p, nrow=p);
    
    # --- Center data ---
    x <- sweep(x, 2, colMeans(x), '-');
    
    # --- Shrinkage intensity ---
    if(missing(alpha)) {
        # -- Phi estimator --
        y <- x^2;
        y <- sweep(y, 1, sqrt(w), '*');
        phi.mat <- crossprod(y)/(1-sum(w^2)) - S^2;
        phi <- sum(phi.mat);
        
        # -- Gamma estimator --
        gamma <- norm(S-T, 'F')^2;
        
        # -- Shrinkage constant --
        kappa <- phi/gamma;
        alpha <- max(0, min(1, kappa/(N-1)));
    }
    
    # --- Shrinkage estimator ---
    Sig <- alpha*T + (1-alpha)*S;
    
    return( list(cov=Sig, shrinkage=alpha) )
}

# --- Stein-type shirinkage with OAS penalty estimator ---
oasCov <- function(x, alpha, w) {
    # --- Data dimension ---
    N <- nrow(x);
    p <- ncol(x);
    
    # --- Sample estimate ---
    if(!missing(w)) {
        S <- cov.wt(x, wt=w)$cov;
    } else {
        S <- cov(x);
    }
    
    # --- Target estimator ---
    var.mu <- mean(diag(S));
    T <- diag(var.mu, ncol=p, nrow=p);
    
    # --- Shrinkage intensity ---
    if(missing(alpha)) {
        # -- Numerator --
        over.mean <- mean(S^2);
        num <- over.mean + var.mu^2;
        
        # -- Denominator --
        denom <- (N + 1)*(over.mean-(var.mu^2)/p);
        
        # -- Parameter --
        alpha <- ifelse(denom==0, 1, min(num/denom,1));
    }
    
    # --- Shrinkage estimator ---
    Sig <- alpha*T + (1-alpha)*S;
    # Sig <- diag(diag(Sig), ncol=ncol(x), nrow=ncol(x));
    
    return( list(cov=Sig, shrinkage=alpha) )
}

# --- Schafer-Strimmer estimator with Ledoit-Wolf shrinkage intensity ---
ssCov <- function(x, w) {
    # -- Shrinkage estimator --
    options(warn=-1)
    if(!missing(w)) {
        S <- corpcor::cov.shrink(x, w=w, verbose=FALSE);
    } else {
        S <- corpcor::cov.shrink(x, verbose=FALSE);
    }
    options(warn=0)
    
    # -- Numeric matrix --
    Sig <- matrix(as.numeric(S), ncol=ncol(x));
    
    return( list(cov=Sig, shrinkage=attr(S,'lambda')) )
}

# --- Minka's (2000) Laplace evidence ---
prinDimLaplaceEvidence <- function(k, p, N, e) {
    # k: principal components dimension
    # p: data dimension
    # N: sample size
    # e: sample eigenvalues
    
    # --- Impose a constraint over the p-k smallest eigenvalues ---
    s.ml <- sum(e[(k+1):p])/(p-k);
    s.ml <- ifelse(is.finite(s.ml), s.ml, 0);
    e[(k+1):p] <- s.ml;
    
    # --- Posterior probability ---
    u.prior.log <- -k*log(2) + sum( lgamma((p-1:k+1)/2) ) -
        0.5*log(pi)*sum(p-1:k+1);
    az.det.log <- sapply(1:k, function(k, e, p, N) {
        sum(log(e[(k+1):p]^(-1)-e[k]^(-1)) + log(e[k]-e[(k+1):p]) + log(N))},
        e=e, p=p, N=N, simplify='array');
    az.det.log <- sum(az.det.log);
    s.ml <- sum(e[(k+1):p])/(p-k);
    n.fp <- p*(p-1)/2 - (p-k)*(p-k-1)/2; # or p*k-k*(k-1)/2
    post.prob.log <- 1.5*k*log(2) + u.prior.log - 0.5*N*sum(log(e[1:k])) - 
        0.5*N*(p-k)*log(s.ml) + 0.5*(n.fp+k)*log(2*pi) - 0.5*az.det.log -
        0.5*k*log(N);
    
    # --- Return the negative log-posterior ---
    return(-post.prob.log)
}

# --- Probabilistic PCA estimator ---
ppcaCov <- function(x, k) {
    # --- Data dimension ---
    N <- nrow(x);
    p <- ncol(x);
    max.vec <- min(p, N-1);
    
    # --- Center data ---
    x <- sweep(x, 2, colMeans(x), '-');
    
    # --- SVD ---
    S.svd <- svd(x, nu=0, nv=max.vec);
    samp.val <- numeric(p);
    samp.val[1:max.vec] <- S.svd$d[1:max.vec]^2/(N-1);
    samp.vec <- S.svd$v;
    
    # --- Choose PC's ---
    if(missing(k)) {
        k <- 1:(max.vec-1);
        k.neg.loglik <- sapply(k, prinDimLaplaceEvidence, p, N, samp.val, 
                               simplify='array');
        k.best <- k[which.min(k.neg.loglik)];
    } else {
        k.best <- k;
    }
    
    # --- k-rank estimator ---
    s.ml <- sum(samp.val[(k.best+1):p])/(p-k.best);
    w.ml <- sqrt( diag(samp.val[1:k.best], ncol=k.best, nrow=k.best) -
                      diag(s.ml, ncol=k.best, nrow=k.best) );
    w.ml <- samp.vec[, 1:k.best, drop=FALSE]%*%w.ml;
    Sig <- diag(s.ml, ncol=p, nrow=p) + tcrossprod(w.ml);
    
    return( list(cov=Sig, rank=k.best, constraint=s.ml) )
}

# --- Sparse Probabilistic PCA estimator ---
spcCov <- function(x, k) {
    
    # --- Data dimension ---
    N <- nrow(x);
    p <- ncol(x);
    max.vec <- min(p, N-1);
    
    # --- Center data ---
    x.means <- colMeans(x);
    x <- sweep(x, 2, x.means, '-');
    
    # --- SSVD ---
    reg.path <- seq(from=1.1, to=sqrt(max.vec), length=100);
    lambda   <- 1 - oasCov(x)$shrinkage;
    best.sumabsv <- quantile(reg.path, lambda);
    # 5-fold cross-validation
    # cv.out <- SPC.cv(x, sumabsvs=reg.path, nfolds=5, niter=10, trace=FALSE,
                     # orth=TRUE, center=TRUE);
    S.ssvd <- PMA::SPC(x, sumabsv=best.sumabsv, niter=25, K=max.vec, 
                  orth=TRUE, trace=FALSE, center=TRUE, compute.pve=FALSE);
    sparse.val <- numeric(p);
    sparse.val[1:max.vec] <- S.ssvd$d[1:max.vec]^2/(N-1);
    sparse.vec <- S.ssvd$v;
    
    # --- Choose PC's ---
    if(missing(k)) {
        k <- 1:(max.vec-1);
        samp.val <- numeric(p);
        samp.val[1:max.vec] <- svd(x, nu=0, nv=0)$d[1:max.vec]^2/(N-1);
        k.neg.loglik <- sapply(k, prinDimLaplaceEvidence, p, N, samp.val, 
                               simplify='array');
        k.best <- k[which.min(k.neg.loglik)];
    } else {
        k.best <- k;
    }
    
    # --- k-rank estimator ---
    s.ml <- sum(sparse.val[(k.best+1):p])/(p-k.best);
    w.ml <- sqrt( diag(sparse.val[1:k.best], ncol=k.best, nrow=k.best) -
                      diag(s.ml, ncol=k.best, nrow=k.best) );
    w.ml <- sparse.vec[, 1:k.best, drop=FALSE]%*%w.ml;
    Sig <- diag(s.ml, ncol=p, nrow=p) + tcrossprod(w.ml);
    
    return( list(cov=Sig, rank=k.best, constraint=s.ml) )
}

# --- Quadratic Glasso ---
quicCov <- function(x, alpha, w) {
    
    # --- Sample estimate ---
    if(!missing(w)) {
        S <- cov.wt(x, wt=w)$cov;
    } else {
        S <- cov(x);
    }
    
    # --- Sparse estimator ---
    if(missing(alpha)) {
        # -- Sparsity level --
        S.max <- abs(max(S));
        rho.path <- seq(1e-1, 1e-2, length=10);
        
        # -- Cross-velidation --
        # [Banerjee - 2008] Model Selection Through Sparse Maximum Likelihood Estimation
        # Gap = -logdetW - p - logdetX + trSX + l1normX;
        gl.model <- QUIC::QUIC(S, rho=S.max, path=rho.path, msg=0);
        Sig <- gl.model$W[,,which.max(gl.model$dGap)];
        alpha <- S.max*rho.path[which.max(gl.model$dGap)];
    } else {
        Sig <- QUIC::QUIC(S, rho=alpha, msg=0)$W;
    }
    return( list(cov=Sig, penalty=alpha) )
}