function [data, labels] = mogdata(N, D, K)
    N_d = numperclass(N, K);
    data = zeros(D, N);
    labels = zeros(N, 1);
    
    idx = zeros(2, 1);
    m = zeros(D, 1);
    lmd = 1e-2;
    mu = D + 3;
    
    for k=1:K
        idx(1) = idx(2) + 1;
        idx(2) = idx(1) + N_d(k) - 1;
        SIGMA = wishrnd(diag(ones(D, 1)), mu);
        MU = mvnrnd(m, inv(lmd.*SIGMA));
        data(:, idx(1):idx(2)) = mvnrnd(MU, inv(SIGMA), N_d(k))';
        labels(idx(1):idx(2)) = k;
    end
end

function N_d = numperclass(N, K)
    N_d = rand(K, 1);
    N_d = round(N_d./sum(N_d) * N);
    res = sum(N_d) - N;
    [~, i] = max(N_d);
    N_d(i) = N_d(i) - res;
end

function numperclassTest(N, K)
    for i=1:1000
        N_d = numperclass(N, K);
        assert(sum(N_d) == N, 'numbers of samples per class do not agree');
    end
end