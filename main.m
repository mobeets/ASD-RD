%% init
X = load('stim.mat');
S = X.stim;
Sxy = X.xy;

% S = S(60:end, :, :); % looks weird in the first 60
S = stim(10000, size(S,2), size(S,3));

Sb = reshape(S, size(S,1), size(S,2)*size(S,3)); % reshape
Ss = squeeze(sum(S, 2)); % sum across time
St = sum(S, 3); % sum across space

%% response
[R, D, ws, wt] = resp(S, Sxy);
% plotX('xy', Sxy, ws); % show space weights
rmse = @(a, b) sqrt((a-b)'*(a-b)); % for assessing fits
results = @(Rh, msg) ['rmse (' msg ') = ' num2str(rmse(R, Rh))];

%% fit time, space, each ignoring the other
[Rhb, whb] = linreg(Sb, R);
Wb = reshape(whb, size(S,2), size(S,3));
[Rhs, whs] = linreg(Ss, R);
[Rht, wht] = linreg(St, R);

%% fit
[RhASD, whASD] = ASD(Ss, R, D.^2);

%% results

disp(results(Rht, 'ols - time'));
disp(results(Rhb, 'ols - both'));
disp(results(Rhs, 'ols - space'));
disp(results(RhASD, 'ASD - space'));
