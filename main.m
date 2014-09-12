%% init
X = load('stim.mat');
S = X.stim;
Sxy = X.xy;
nt = size(S, 2);
ns = size(S, 3);

% S = S(60:end, :, :); % looks weird in the first 60
N = 10000;
S = stim(N, nt, ns);

Sb = reshape(S, N, nt*ns); % reshape
Ss = squeeze(sum(S, 2)); % sum across time
St = sum(S, 3); % sum across space

%% response
noise_sigma = 1;
[R, D, ws, wt] = resp(S, Sxy, noise_sigma);
plotX('xy', Sxy, ws); % show space weights
rmse = @(a, b) sqrt((a-b)'*(a-b)); % for assessing fits
results = @(Rh, msg) ['rmse (' msg ') = ' num2str(rmse(R, Rh))];

%% fit time, space, each ignoring the other
[Rhb, whb] = linreg(Sb, R);
Wb = reshape(whb, nt, ns);
[Rhs, whs] = linreg(Ss, R);
[Rht, wht] = linreg(St, R);
[Rhr, whr] = ridge(Ss, R);
%
% ridge regression solution no different from ASD
%

%% fit
[RhASD, whASD] = ASD(Ss, R, D.^2);
%
% problems with prior/posterior covariance having negative eigenvalues
% is usually fixed by shrinking the bounds on p, the scale parameter
% in general, the more noise in the response, the smaller the allowable
% range for p.
% 
% n.b. sigma_noise in resp ? ssq estimate, i.e. theta(1)
% and as ssq increases, the scale parameter p tends to shrink towards 0.
% 

%% results

disp(results(Rht, 'ols - time'));
disp(results(Rhb, 'ols - both'));
disp(results(Rhs, 'ols - space'));
disp(results(Rhr, 'ridge - space'));
disp(results(RhASD, 'ASD - space'));

%% plot

plotX('xy', Sxy, ws); title('REAL');
plotX('xy', Sxy, whASD); title('ASD');
plotX('xy', Sxy, whr); title('ridge');
plotX('xy', Sxy, whs); title('ols');
