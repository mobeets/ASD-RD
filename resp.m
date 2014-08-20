function [R, D, ws, wt] = resp(S, pts)
    [n, nt, ns] = size(S);
    [D, ~, ws] = randomDistancesGaussianWeights(pts);
    
    % history filter
    x = 1:nt;
    wt_fcn = @(k, th) (x(:).^(k-1)).*exp(-x(:)/th);
    wt = wt_fcn(5, 1);

    sigma = 1;
    R = zeros(n, 1);
    for ii = 1:n
        R(ii) = wt'*squeeze(S(ii,:,:))*ws;
    end
    R = R + sigma*randn(n, 1);

end
