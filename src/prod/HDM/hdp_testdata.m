% For testing
% K = 10; J = 30; V = 100; eta = [1.5 1.2]; alpha = [3 2.5 4 4]; beta = 0.3; n_j_min = 80; n_j_max = 120;
function hdp_testdata(K, J, V, eta, alpha, beta, n_j_min, n_j_max)
    z = zeros(J, 1);
    
    z = ncrp(eta, z);
    pi = sbptree(alpha, K, z);
    
    beta = repmat(beta, V, 1);
    phi = zeros(K, V);
    for k=1:K
        phi(k, :) = dirichletrnd(beta);
    end
    
    w = cell(J, 1);
    n_j_rand = n_j_max - n_j_min;
    for j=1:J
        n_j = n_j_min + randi(n_j_rand);
        x_j = zeros(n_j, 1);
        theta = pi{end}(:, j);
        
        for i=1:n_j
            k = find(mnrnd(1, theta));
            v = find(mnrnd(1, phi(k, :)));
            x_j(i) = v;
        end
        
        w{j} = x_j;
    end
    
    [perplexity, loglik] = evalperp(w, pi{end}, phi);
    
    groundtruth = struct('pi', {pi}, 'phi', phi, 'z', z, 'perplexity', perplexity, 'loglik', loglik);
    drawdata(groundtruth);
    
    save('testdata_hdp', 'w', 'V', 'groundtruth');
end

function drawdata(groundtruth)
    pi = groundtruth.pi;
    z = groundtruth.z;
    L = length(pi);
    K = size(pi{1}, 1);
    
    N = size(z, 1);
    z = [ones(N, 1) z (1:N)'];
    
    figure(1);
    subplot(1, 1, 1); stem(pi{1}, 'Marker', 'None');
    xlim([0 K+1]);set(gca, 'XTick', []);
    parent = 1;
    
    for l=2:L
        idx = z(:, l - 1) == parent;
        z_idx = unique(z(idx, l));
        
        num = 4;
        numpi = length(z_idx);
        if numpi < num
            num = numpi;
        end
        
        idx = randperm(numpi, num);
        nextparent = z_idx(idx);
        
        figure(l);
        for i=1:num
            subplot(num, 1, i); stem(pi{l}(:, nextparent(i)), 'Marker', 'None');
            xlim([0 K+1]);set(gca, 'XTick', []);
        end
        
        parent = nextparent(1);
    end
end

    
    
    
    