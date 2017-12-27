% nested Chinese Restaurant Process by truncated stick-breaking process
% sum(pi(:,1)) should be L (sum of probabilities placed in L levels of partitions)

function pi = ncrp(gamma, T, N)
    L = length(T);
    
    children = zeros(1, L);
    children(1) = T(1);
    for l=2:L
        children(l) = children(l - 1) * T(l);
    end
    numnode = sum(children);
    
    pi = zeros(numnode, N);
    i = 1;
    pi_parent = ones(1, N);
    
    for l=1:L
        pi_tmp = helper(gamma, T, N, children, l, pi_parent);
        
        pi(i:i+children(l)-1, :) = pi_tmp;
        i = i + children(l);
        
        pi_parent = pi_tmp;
    end
end

function pi_tmp = helper(gamma, T, N, children, l, pi_parent)
    itr = children(l)/T(l);
    pi_tmp = zeros(children(l), N);
    j=1;
    for i=1:itr
        pi_tmp(j:j+T(l)-1, :) = tsbp(gamma, T(l), N) .* repmat(pi_parent(i, :), T(l), 1);
        j = j + T(l);
    end
end