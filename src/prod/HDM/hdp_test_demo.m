%% generate test data
K = 3;
J = 10;
V = 100;
gamma = 1;
alpha = 10;
beta = 0.3;
n_j_min = 100;
n_j_max = 150;
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
beta      = 0.1;  % parameter of dirichlet distribution (symmetric)
maxitr    = 20;
maxitr_hp = 50;
logfile = 'hdp_demo.log';

tic;
hdp_test('testdata_hdp.mat', logfile, gamma, a_gam, b_gam, alpha, a_alpha, b_alpha, beta, maxitr, maxitr_hp);
t = toc;
fprintf('%3.3f seconds elapsed\n', t)

tic;
hdp_test('testdata_hdp.mat', logfile, gamma, a_gam, b_gam, alpha, a_alpha, b_alpha, beta, maxitr, maxitr_hp);
t = toc;
fprintf('%3.3f seconds elapsed\n', t)

tic;
hdp_test('testdata_hdp.mat', logfile, gamma, a_gam, b_gam, alpha, a_alpha, b_alpha, beta*1e-1, maxitr, maxitr_hp);
t = toc;
fprintf('%3.3f seconds elapsed\n', t)

diary off