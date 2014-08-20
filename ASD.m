function [Rh wh] = ASD(S, R, D)
    % optimizes log evidence to find hyperparameters
    t0 = [1.0 1.0 1.0];
    sta = S'*R;
    stim_cov = S'*S;
%     [mu, cov] = posterior_mean_and_cov(stim_cov, sta, D, t0);

%     objfcn = @(theta) ASD_logEvidence(theta, S, R, mu, cov, D);
    objfcn = @(theta) ASD_logEvidence(theta, S, R, stim_cov, sta, D);
    options = optimoptions(@fmincon, 'GradObj', 'on');
    % old lb,ub for theta(3) = 1e-5, 1e7
    theta = fmincon(objfcn, t0, [], [], [], [], [-2 1e-5 1e-5], [20 1e5 2.05], [], options);

    [mu, cov] = posterior_mean_and_cov(stim_cov, sta, D, theta);
    Reg = ASD_Regularizer(theta(2:end), D, mu, cov);
    wh = (S'*S + Reg)\(S'*R);
    Rh = S*wh;
end

function [mu, cov] = posterior_mean_and_cov(stim_cov, sta, D, theta)
    % update posterior mean and covariance
    cov = inv(stim_cov/theta(1) + ASD_Regularizer(theta(2:end), D));
    mu = cov*sta/theta(1);
end

% function [v, dv] = ASD_logEvidence(theta, X, Y, mu, cov, D)
function [v, dv] = ASD_logEvidence(theta, X, Y, stim_cov, sta, D)
    [mu, cov] = posterior_mean_and_cov(stim_cov, sta, D, theta);
    [C, dC] = ASD_Regularizer(theta(2:end), D, mu, cov);
    v = -logE(C, theta(1), cov, X, Y);
    if nargout > 1
        dssq = dlogE_dssq(C, theta(1), X, Y, mu, cov);
        dv = -[dssq dC];
    end
end

function v = logE(C, sig, cov, X, Y)
    n = size(cov, 1);
    logDet = @(A) 2*sum(diag(chol(A)));
    z1 = 2*pi*cov;
    z2 = 2*pi*sig^2*eye(n, n);
    z3 = 2*pi*C;
    try
        logZ = 0.5*(logDet(z1) - (logDet(z2) + logDet(z3)));
    catch err
        disp('-----ERROR-----');
        if sum(eig(cov) < 0) > 0
            %   * usually because cov not being regularized enough
            disp('cov has negative eigenvalues.');
        end
        if sum(eig(C) < 0) > 0
            %   * usually because C is all zeros
            disp('Regularizer has negative eigenvalues.');
        end
        1;
    end
    B = (1/sig^2) - (X*cov*X')/sig^4;
    v = logZ - 0.5*Y'*B*Y;
end

function v = dlogE_dssq(C, sig, X, Y, mu, cov)
    T = numel(Y);
    n = size(cov, 1);
    V1 = eye(n, n) - cov\C;
    V2 = (Y - X*mu)'*(Y - X*mu);
    V = -T + trace(V1) + V2/sig^2;
    v = V/sig^2;
end
