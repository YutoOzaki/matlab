%% Generate random Dirichlet variables
% For example, compare how sparsity of distribution changes with parameters
% subplot(411);plot(dirichletrnd(1.0   .* ones(1000,1))); title('beta = 1.0');
% subplot(412);plot(dirichletrnd(0.1   .* ones(1000,1))); title('beta = 0.1');
% subplot(413);plot(dirichletrnd(0.01  .* ones(1000,1))); title('beta = 0.01');
% subplot(414);plot(dirichletrnd(0.001 .* ones(1000,1))); title('beta = 0.001');
function x = dirichletrnd(alpha)
    x = gamrnd(alpha, 1);
    x = x./sum(x);
end