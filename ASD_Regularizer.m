function [C, dC] = ASD_Regularizer(theta, Ds, mu, cov)
    if nargin < 1
        [cov, mu, p, ds, Ds] = defaultValues();
    else
        p = theta(1);
        ds = theta(2:end);
    end
%     for ii = 1:numel(ds)
%         D = Ds(:,:,ii);
%         assert(isequal(D, D'), 'D must be symmetric.');
%         assert(isequal(diag(D), zeros(size(D, 1), 1)), 'D must have a zero diagonal.');
%     end
%     C = exp(-p - 0.5*(Ds(:,:,1)/ds(1)^2));
    C = exp(-p - Ds/ds^2);
%     assert(all(eig(C) > 0), 'C is not positive definite--check your distance matrices.');

    if nargout > 1
        dC = gradient(C, mu, cov, Ds, ds);
    end
end

function dC = gradient(C, mu, cov, Ds, ds)
    A = (C - cov - mu*mu')/C;
%     dC = zeros(1, numel(ds)+1);
%     dC(:,1) = 0.5*trace(A);
%     for ii = 1:numel(ds)
%         dC(:,ii+1) = -0.5*trace(A*(C .* Ds(:,:,ii)/(ds(ii)^3))/C);
%     end
    dC = [0.5*trace(A) -0.5*trace(A*(C .* Ds(:,:,1)/(ds(1)^3))/C)];
end

function [mu, cov, p, ds, Ds] = defaultValues(nD)
    ds = ones(nd, 1);
    p = -1;
    n = 99;
    cov = eye(n, n);
    mu = randn(n, 1);
    successFcn = @(D) all(eig(exp(1 - 0.5*D)) > 0);
    Ds = randomDistances(n, nD, successFcn);
end
