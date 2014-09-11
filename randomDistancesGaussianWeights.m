function [D, pts, w] = randomDistancesGaussianWeights(pts, mu, Sigma, nw, b, n)
    % returns a square (nw-by-nw) matrix of distances between randomly
    % selected 2d positions, and a vector of weights, where the weight of a
    % point is given by the pdf of a 2d gaussian at that location.
    %
    % mu and Sigma parameterize the 2d gaussian
    % pts is nw-by-2; if false, pts will be randomly chosen
    % nw is number of weights, if pts is not provided
    % b is bounds
    % n is step size within those bounds
    % nw is number of positions to be drawn

    defarg('pts', false);
    defarg('nw', 50);
    defarg('b', 5);
    defarg('n', 100);

    if size(pts, 2) ~= 2
        pts = randomPoints(nw, b, n);
    end
    
    defarg('mu', mean(pts) + 1);
    defarg('Sigma', std(pts)/1.2);
    
    w = mvnpdf(pts, mu, Sigma);
    D = squareform(pdist(pts, 'euclidean'));
end

function pts = randomPoints(nw, b, n)
    xi = linspace(-b, b, n);
    yi = xi;
    idx = randi(n, 2, nw);
    x = xi(idx(1,:));
    y = yi(idx(2,:));
    pts = [x; y]';
end
