%% generate test data
K = 10;
J = 30;
V = 100;
gamma = 2;
alpha = 2;
beta = 0.4;
n_j_min = 80;
n_j_max = 120;
converged = 10;
INCREMENTAL = false;

hdp_testdata(K, J, V, gamma, alpha, beta, n_j_min, n_j_max, converged, INCREMENTAL);

%% run hdp
gamma     = 1;
a_gam     = 1;    % Shape parameter of gamma prior for gamma
b_gam     = 0.1;    % Scale parameter of gamma prior for gamma
alpha     = 1;
a_alpha   = 1;    % Shape parameter of gamma prior for alpha
b_alpha   = 1;    % Scale parameter of gamma prior for alpha
beta      = [0.1 0.1 0.1 0.1 0.1 0.1];  % parameter of dirichlet distribution (symmetric)

steps     = [100 100   1   1   1   1];
steps_hp  = [ 50  50  50  50   0   0];
steps_fpi = [100   0 100   0 100   0]; % fixed point iteration
maxitr    = [ 20  20 600 600 600 600];
scheme    = {'direct', 'direct', 'direct', 'direct', 'direct', 'direct'}; % chinese restaurant franchise, direct assignment, and posterior representation sampler

trial = 6;
assert(trial == length(beta) && trial == length(steps) && trial == length(steps_hp)...
    && trial == length(maxitr) && trial == length(scheme) && trial == length(steps_fpi),...
    'wrong setup for hyperparameters');

%% Setup logging
if strcmp(computer, 'PCWIN64')
    slash = '\';
elseif strcmp(computer, 'MACI64')
    slash = '/';
end

homedir = userpath;
homedir = homedir(1:(length(homedir) - 1));
logging = strcat(homedir,slash,'logs',slash,'HDM',slash,'hdp_demo.log');
diary(logging);
fprintf('\n*** Experiment %5.5f ***\n', now);

%% Experiment
for i=1:trial
    fprintf('MCMC steps of parameters = %d\n', steps(i));
    fprintf('MCMC steps of hyperparameters = %d\n', steps_hp(i));
    fprintf('steps of fixed-point iteration = %d\n', steps_fpi(i));
    fprintf('number of iterations = %d\n', maxitr(i));
    fprintf('posterior inference scheme = %s\n', scheme{i});
    
    tic;
    hdp_test('testdata_hdp.mat', gamma, a_gam, b_gam, alpha, a_alpha, b_alpha,...
        beta(i), steps(i), steps_hp(i), steps_fpi(i), maxitr(i), scheme{i});
    t = toc;
    fprintf('%3.3f seconds elapsed\n\n', t)
end

fprintf('done!\n');

diary off