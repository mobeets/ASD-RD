function plotX(cmd, x, y)
    figure;
    switch cmd
        case 'xy'
            if nargin < 3 || size(x,1) ~= numel(y)
                y = 100*ones(size(x,1), 1);
            else
                y = round(1e3*y);
                y = y - min(y) + 1;
            end
            scatter(x(:,1), x(:,2), y, 'filled', 'k');
        case 'w'
            plot(x, 'o');
        case 's'
            scatter(x, y);
        case 'im'
            colormap(gray); imagesc(x);
    end
end
