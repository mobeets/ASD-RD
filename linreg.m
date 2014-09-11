function [Rh, wh] = linreg(S, R)
    wh = ols(S, R);
    Rh = S*wh;
end

function Rh = ols(x, y)
% solution to ordinary least squares regression of y on x
    Rh = (x'*x)\(x'*y);
end
