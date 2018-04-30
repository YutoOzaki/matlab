function main_MNIST
    loadMNIST();
    datafile = 'mnistdata.mat';
    K = 10;
    gpumode = true;
    
    %{
    numepoch = 5;
    h = [500, 2000, 10];
    act = {relutrans(), relutrans()};
    L = 1;
    pretraining_rpsae(datafile, h, act, L, numepoch, gpumode, @reconMNIST);
    clf
    %}
    
    %{
    numepoch = 10;
    L = 1;
    fitting_kmeans(datafile, K, L, numepoch, gpumode, @disptsne, @displabel);
    clf
    %}
    
    %{
    numepoch = 5;
    L = 1;
    takeover = true;
    fitting_sgd(datafile, K, L, numepoch, gpumode, takeover, @disptsne, @displabel);
    clf
    %}
    
    numepoch = 3;
    L = 1;
    userscript(datafile, L, numepoch, gpumode, @reconMNIST, @disptsne, @displabel);
    clf
    
    load('vde_clustering_result.mat');
    
end

function reconMNIST(x, output)
    I = 10;
    
    for i=1:I
        subplot(I, 2, i);
        drawMNIST(x(:, i));
        
        subplot(I, 2, I+i);
        drawMNIST(output(:, i));
    end
end

function disptsne(rprsn, labels, epoch)
    if rem(epoch, 15) == 0
        dim = 2;
        pcadim = 5;
        perp = 50;
        theta = 0.5;
        alg = 'svd';

        if isa(rprsn, 'gpuArray')
            rprsn = gather(rprsn);
        end
        
        map = fast_tsne(rprsn', dim, pcadim, perp, theta, alg);
        gscatter(map(:, 1), map(:, 2), labels);
    end
end

function displabel(x, labels)
    r = 10;
    c = 2;
    idx = randperm(size(x, 2));
    
    labels_buf = labels(idx(1:r*c));
    [~, I] = sort(labels_buf);
    
    for i=1:(c*r)
        subplot(r, c, i);
        
        idx_i = I(i);
        drawMNIST(x(:, idx(idx_i))); title(num2str(labels(idx(idx_i))));
    end
end