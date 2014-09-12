function [Rh, wh, ssq, theta] = ridge(S, R)
    % finds ridge regularier hyperparameter using fixed-point algorithm
    % source: Park, Pillow (2011) Methods

    sta = S'*R;
    stim_cov = S'*S;
    [Rh, ~] = linreg(S, R); % ML estimate
    errs = R - Rh;
    mse = errs'*errs / numel(R);

    % find mean
    t0 = [1e-6 mse]; % initial guess
    [ssq, theta] = fixed_point(S, R, sta, stim_cov, t0);
    % or can i use posterior mean as my estimate of wh?
    [mu, ~] = posterior_mean_and_cov(stim_cov, sta, theta, ssq, size(S, 2));
%     invReg = invregularizer(theta, size(S,2));
    wh = mu; %(S'*S + invReg)\(S'*R);
    Rh = S*wh;
end

function Reg = invregularizer(lambda, d)
    Reg = diag(lambda*ones(d, 1));
end

function [mu, post_cov] = posterior_mean_and_cov(stim_cov, sta, theta, sigmasq, d)
    % update posterior mean and covariance
    post_cov = inv(stim_cov/sigmasq + invregularizer(theta, d));
    mu = post_cov*sta/sigmasq;
end

function [new_theta, new_sigmasq] = ridge_update(theta, sigmasq, n, d, stim_cov, sta, S, R)
    [mu, post_cov] = posterior_mean_and_cov(stim_cov, sta, theta, sigmasq, d);
    new_theta = (d - theta*trace(post_cov))/(mu'*mu);
	errs = (R - S*mu)';
	new_sigmasq = (errs*errs')/(n - d + theta*trace(post_cov));
end

function [s1, t1] = fixed_point(S, R, sta, stim_cov, t0)
    % controls # of iterations
    tol = 1e-10;
    maxiters = 1000;
    
    % variables reused
    n = numel(R);
    d = size(S, 2); % parameter dimensionality
    
    sigmasqs = nan(maxiters+1, 1);
    thetas = nan(maxiters+1, 1);
    thetas(1) = t0(1);
    sigmasqs(1) = t0(2);
    for ii=1:maxiters
        tc = thetas(ii);
        sc = sigmasqs(ii);
        [t1, s1] =  ridge_update(tc, sc, n, d, stim_cov, sta, S, R);
        thetas(ii+1,:) = t1;
        sigmasqs(ii+1) = s1;
        % stop if changes in sigmasq update is within tolerance
        if abs(s1 - sc) < tol
            break;
        end
    end
end
