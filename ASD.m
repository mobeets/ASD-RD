function [Rh, wh] = ASD(S, R, D)
    % optimizes log evidence to find hyperparameters
    t0 = [1.0 1.0 1.0];
    sta = S'*R;
    stim_cov = S'*S;

    objfcn = @(theta) ASD_logEvidence(theta, S, R, stim_cov, sta, D);
    options = optimoptions(@fmincon, 'GradObj', 'on');
    % old lb,ub for theta(3) = 1e-5, 1e7
    lb = [-20 1e-5 1e-5];
    ub = [20 1e5 1e7];
    theta = fmincon(objfcn, t0, [], [], [], [], lb, ub, [], options);

    [mu, post_cov] = posterior_mean_and_cov(stim_cov, sta, D, theta);
    Reg = ASD_Regularizer(theta(2:end), D, mu, post_cov);
    wh = (S'*S + Reg)\(S'*R);
    Rh = S*wh;
end

function [mu, post_cov] = posterior_mean_and_cov(stim_cov, sta, D, theta)
    % update posterior mean and covariance
    post_cov = inv(stim_cov/theta(1) + ASD_Regularizer(theta(2:end), D));
    mu = post_cov*sta/theta(1);
end

function [v, dv] = ASD_logEvidence(theta, X, Y, stim_cov, sta, D)
    [mu, post_cov] = posterior_mean_and_cov(stim_cov, sta, D, theta);
    [C, dC] = ASD_Regularizer(theta(2:end), D, mu, post_cov);
    v = -logE(C, theta(1), post_cov, X, Y);
    if nargout > 1
        dssq = dlogE_dssq(C, theta(1), X, Y, mu, post_cov);
        dv = -[dssq dC];
    end
end

function v = logE(C, sig, post_cov, X, Y)
    n = size(post_cov, 1);
    logDet = @(A) 2*sum(diag(chol(A)));
    z1 = 2*pi*post_cov;
    z2 = 2*pi*sig^2*eye(n, n);
    z3 = 2*pi*C;
    try
        logZ = 0.5*(logDet(z1) - (logDet(z2) + logDet(z3)));
    catch err
        disp('-----ERROR-----');
        if sum(eig(post_cov) < 0) > 0
            %   * usually because post_cov not being regularized enough
            disp('post_cov has negative eigenvalues.');
        end
        if sum(eig(C) < 0) > 0
            %   * usually because C is all zeros
            disp('Regularizer has negative eigenvalues.');
        end
    end
    B = (1/sig^2) - (X*post_cov*X')/sig^4;
    v = logZ - 0.5*Y'*B*Y;
end

function v = dlogE_dssq(C, sig, X, Y, mu, post_cov)
    T = numel(Y);
    n = size(post_cov, 1);
    V1 = eye(n, n) - post_cov\C;
    V2 = (Y - X*mu)'*(Y - X*mu);
    V = -T + trace(V1) + V2/sig^2;
    v = V/sig^2;
end
