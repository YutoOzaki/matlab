function [pi, repo] = directassignmentWrapper(x, alpha, pi, z, f_k, repo, steps)
    if nargin == 5
        repo = [];
        steps = 0;
    elseif nargin == 6
        steps = 0;
    end
    
    if isempty(z)
        G = 1;
        z_buf = ones(length(x), 1);
    else
        G = length(unique(z(:, end)));
        z_buf = z(:, end);
    end
    
    mysbp = @(pi) stickbreaking(alpha, pi, z);
    [pi, n_jk, n_kv, n_k, z_ji] = directassignment(x, alpha(end), pi, f_k, mysbp, z_buf, repo, steps);
    
    idxSet = cell(G, 1);
    n_jkSet = cell(G, 1);
    
    for g=1:G
        idx = z_buf == g;
        
        idxSet{g} = find(idx);
        n_jkSet{g} = n_jk(idx, :);
    end
    
    repo = struct('n_kv', n_kv, 'n_k', n_k, 'n_jk', n_jk, 'z_ji', {z_ji}, 'idxSet', {idxSet}, 'n_jkSet', {n_jkSet});
end