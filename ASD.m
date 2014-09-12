function [Rh, wh] = ASD(S, R, D)
    % optimizes log evidence to find hyperparameters
    sta = S'*R;
    stim_cov = S'*S;
    objfcn = @(theta) ASD_logEvidence(theta, S, R, stim_cov, sta, D);

    [~, ~, sigmasq, theta] = ridge(S, R);
    t0 = [sigmasq -log(theta) 1.0];
%     t0 = [1.0 1.0 1.0];
    t0
    lb = [10e-6 -5 10e-6];
    ub = [10e6 5 10e6];
    assert(all(lb <= t0 & t0 <= ub));
    options = optimoptions(@fmincon, 'GradObj', 'on');
    theta = fmincon(objfcn, t0, [], [], [], [], lb, ub, [], options);
    
    theta
    
    C = ASD_Regularizer(theta(2:end), D);
    [mu, post_cov] = posterior_mean_and_cov(stim_cov, sta, C, theta(1));
    % or do I just use mu as my wh estimate??
    wh = mu; % (stim_cov + inv(C))\sta;
    Rh = S*wh;
end

function [mu, post_cov] = posterior_mean_and_cov(stim_cov, sta, C, ssq)
    % update posterior mean and covariance
    post_cov = inv(stim_cov/ssq + inv(C));
    mu = post_cov*sta/ssq;
end

function [v, dv] = ASD_logEvidence(theta, X, Y, stim_cov, sta, D)
    C = ASD_Regularizer(theta(2:end), D);
    [mu, post_cov] = posterior_mean_and_cov(stim_cov, sta, C, theta(1));
    v = -logE(C, theta(1), post_cov, X, Y);
    if nargout > 1
        dssq = dlogE_dssq(C, theta(1), X, Y, mu, post_cov);
        dC = dlogE_dds(C, theta(3:end), D, mu, post_cov);
        dv = -[dssq dC];
    end
end

function v = logE(C, ssq, post_cov, X, Y)
    n = size(post_cov, 1);
    logDet = @(A) 2*sum(diag(chol(A)));
    z1 = 2*pi*post_cov;
    z2 = 2*pi*ssq*eye(n, n);
    z3 = 2*pi*C;
    try
        logZ = logDet(z1) - logDet(z2) - logDet(z3);
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
        ssq
    end
    B = (eye(size(X,1), size(X,1))/ssq) - (X*post_cov*X')/ssq^2;
    v = 0.5*(logZ - Y'*B*Y);
end

function de = dlogE_dds(C, ds, Ds, mu, post_cov)
    A = (C - post_cov - mu*mu')/C;
%     dC = zeros(1, numel(ds)+1);
%     dC(:,1) = 0.5*trace(A);
%     for ii = 1:numel(ds)
%         dC(:,ii+1) = -0.5*trace(A*(C .* Ds(:,:,ii)/(ds(ii)^3))/C);
%     end
    B = A*(C .* Ds(:,:,1)/(ds(1)^3))/C;
    de = [0.5*trace(A) -0.5*trace(B)];
end

function de = dlogE_dssq(C, ssq, X, Y, mu, post_cov)
    n = size(post_cov, 1);
    V1 = eye(n, n) - post_cov/C;
    sse = (Y - X*mu)'*(Y - X*mu);
    V = -numel(Y) + trace(V1) + sse/ssq;
    de = V/ssq;
end
