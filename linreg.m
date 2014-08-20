function [Rh, wh] = linreg(S, R)
    wh = ols(S, R);
    Rh = S*wh;
end

function Rh = ols(x, y)
    Rh = (x'*x)\(x'*y);
end
