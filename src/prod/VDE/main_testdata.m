function main_testdata
    N = 5000;
    D = 3;
    K = 8;
    datafile = 'testdata.mat';
    
    data = testdata(N, D, K);
    save(datafile, 'data');
    
    numepoch = 30;
    pretraining_rpsae(datafile, numepoch);
    clf
    
    numepoch = 50;
    fitting(datafile, numepoch, K);
    clf
    
    numepoch = 50;
    userscript(numepoch);
    clf
    
    load('vde_clustering_result.mat');
    c = rand(3, K);
    c = bsxfun(@rdivide, c, sum(c, 1));
    for k=1:K
        idx = label == k;
        figure(1);
        scatter3(data(1, idx), data(2, idx), data(3, idx), 20, c(:, k)');hold on
    end
    hold off;
end