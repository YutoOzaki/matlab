function hdplda_demo
    %data = 'testdata_handmade';
    %%{
    K = 10; J = 30; V = 100; eta = [1.5 1.2]; alpha = [3 2.5 4 4]; beta = 0.3; n_j_min = 80; n_j_max = 120;
    hdp_testdata(K, J, V, eta, alpha, beta, n_j_min, n_j_max);
    data = 'testdata_hdp';
    %}
    
    eta = [1.5 1.2]; alpha = [3 2.5 4 4]; beta = 0.01; steps = 20; maxitr = 50;
    [perp1, ~, pi1, ~, ~] = hdplda(data, eta, alpha, beta, steps, maxitr);
    
    eta = [3 2 1]; alpha = [5 4 3 2 1]; beta = 0.01; steps = 20; maxitr = 50;
    [perp2, ~, pi2, ~, ~] = hdplda(data, eta, alpha, beta, steps, maxitr);
    
    eta = []; alpha = [2 3]; beta = 0.01; steps = 20; maxitr = 50;
    [perp3, ~, pi3, ~, ~] = hdplda(data, eta, alpha, beta, steps, maxitr);
    
    eta = [3 4]; alpha = [2 2 4 4]; beta = 0.001; steps = 20; maxitr = 50;
    [perp4, ~, pi4, ~, ~] = hdplda(data, eta, alpha, beta, steps, maxitr);
    
    fprintf('*** result ***\n');
    fprintf('perplexity (4-level hdp): %e\n', min(perp1));
    fprintf('perplexity (5-level hdp): %e\n', min(perp2));
    fprintf('perplexity (2-level hdp): %e\n', min(perp3));
    fprintf('perplexity (4-level hdp): %e\n', min(perp4));
    
    figure(2); imagesc(pi1{end});
    figure(3); imagesc(pi2{end});
    figure(4); imagesc(pi3{end});
    figure(5); imagesc(pi4{end});
        
    try
        load(data, 'groundtruth');
        fprintf('perplexity (ture model) : %e\n', groundtruth.perplexity);
        figure(6); imagesc(groundtruth.pi{end});
    catch e
    end
end