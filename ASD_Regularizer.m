function C = ASD_Regularizer(theta, Ds)
    if nargin < 1
        [p, ds, Ds] = defaultValues();
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
    C = exp(-p - 0.5*Ds/ds^2);
%     assert(all(eig(C) > 0), 'C is not positive definite--check your distance matrices.');

end

function [p, ds, Ds] = defaultValues(nD)
    ds = ones(nd, 1);
    p = -1;
    n = 99;
    successFcn = @(D) all(eig(exp(1 - 0.5*D)) > 0);
    Ds = randomDistances(n, nD, successFcn);
end
