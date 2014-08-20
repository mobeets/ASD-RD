%% init
X = load('data.mat');
S = X.stim.pulses;

%% response
[R, D, ws, wt] = resp(S, X.stim.gaborXY);
rmse = @(a, b) sqrt((a-b)'*(a-b)); % for assessing fits
results = @(Rh, msg) ['rmse (' msg ') = ' num2str(rmse(R, Rh))];

%% fit time, space, each ignoring the other
Sb = reshape(S, size(S,1), size(S,2)*size(S,3)); % reshape
[Rhb, whb] = linreg(Sb, R);
Wb = reshape(whb, size(S,2), size(S,3));

Ss = squeeze(sum(S, 2)); % sum across time
[Rhs, whs] = linreg(Ss, R);

St = sum(S, 3); % sum across space
[Rht, wht] = linreg(St, R);

%% fit
[RhASD, whASD] = ASD(Ss, R, D.^2);

%% results

disp(results(Rht, 'ols - time'));
disp(results(Rhb, 'ols - both'));
disp(results(Rhs, 'ols - space'));
disp(results(RhASD, 'ASD - space'));
