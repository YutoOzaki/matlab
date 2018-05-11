function evalMNIST(datafile, gpumode, K)
    load(strcat(datafile, '_vde_clustering_result.mat'));
    load(datafile);
    
    truelabels = truelabels(:)';
    labels = labels(:)';
    labels = labels - 1;
    if gpumode
        labels = gather(labels);
    end
    
    idx = 0:(K - 1);
    i = zeros(K, 1);
    score = 0;
    for k=1:K
        [~, I] = find(labels == (k - 1));
        
        buf = 0;
        for j=idx
            [~, I_j] = find(truelabels == j);
            
            hit = length(intersect(I, I_j));
            
            if hit > buf
                i(k) = j;
                buf = hit;
            end
        end
        
        score = score + buf;
        idx = setdiff(idx, i(k));
    end
    
    N = length(labels);
    fprintf(' (MNIST result: %3.3f%%)\n', score/N*100);
end