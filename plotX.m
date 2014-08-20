function plotX(cmd, x, y)
    figure;
    switch cmd
        case 'gaborPos'
            if nargin < 3 || size(x,1) ~= numel(y)
                y = 100*ones(size(x,1), 1);
            else
                y = 1e4*y;
            end
            scatter(x(:,1), x(:,2), y, 2, 'filled', 'k');
        case 'w'
            plot(x, 'o');
        case 's'
            scatter(x, y);
        case 'im'
            colormap(gray); imagesc(x);
    end
end
