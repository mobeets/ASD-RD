function [R, D, ws, wt] = resp(S, pts, noise_sigma)
% S is space-time stimulus on each trial
% pts is x,y locations of space as represented in stimulus
% noise_sigma is variance of noise in response
%
% returns the space-time separable weighted response to the given stimulus

    [n, nt, ns] = size(S);
    if nargin < 3
        noise_sigma = 5;
    end
    
    % time weights
    x = 1:nt;
    wt_fcn = @(k, th) (x(:).^(k-1)).*exp(-x(:)/th);
    wt = wt_fcn(5, 1);
    
    % space weights
    [D, ~, ws] = randomDistancesGaussianWeights(pts);
    ws = 5*ws;%*(sum(wt)/sum(ws)); % scale so space and time roughly equal

    R = noise_sigma*randn(n, 1);
    for ii = 1:n
        R(ii) = R(ii) + sum(squeeze(S(ii,:,:)))*ws;
%         R(ii) = R(ii) + wt'*squeeze(S(ii,:,:))*ws;
    end
end
