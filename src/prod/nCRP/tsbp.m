% 1. T is a truncation level of the truncated stick-breaking process.
% 2. v(T) is 1.
% 3. sum(pi) should be 1.
% 4. In variational inference, setting a variational distribution for v(T) is unnecessary since v(T) is 1 with probability 1.

function pi = tsbp(gamma, T, N)
    v = zeros(T, N);
    if T > 1
        v(1:(T-1), :) = betarnd(1, gamma, [T-1, N]);
    end
    v(T, :) = 1;
    pi = zeros(T, N);
    
    remaining = 1.0;
    for i=1:T
        pi(i, :) = v(i, :) .* remaining;
        remaining = remaining .* (1 - v(i, :));
    end
end