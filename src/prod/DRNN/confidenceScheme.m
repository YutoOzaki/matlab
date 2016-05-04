function hit = confidenceScheme(y, patch, labels)
    classNum = size(y, 1);
    N_test = size(y, 2);
    T = size(y, 3);
    e = 1e-8;

    buf = zeros(patch, classNum);
    
    n_start = 1;
    n_end = n_start + patch - 1;
    
    n = 0; hit = 0; logC = log(classNum);
    
    while n_end <= N_test
        n = n + 1;
        
        [~,mind] = max(y(:,n_start:n_end,:));
        mind = squeeze(mind);

        for i=1:classNum
            buf(:,i) = sum((mind./i)==1, 2);
        end
        h = buf'./T + e;

        H = -sum(h.*log(h));
        r = 1 - H./logC;

        c = log(h.*repmat(r,classNum,1)); %log(rates of win * reliability)
        [~,label] = max(sum(c,2));
        
        if label == labels(n)
            hit = hit + 1;
        end
        
        n_start = n_end + 1;
        n_end = n_start + patch - 1;
    end
    
    hit = hit / n;
end