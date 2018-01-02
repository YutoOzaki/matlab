%nCRP-LDA with variational inference
%[2; 1; 0.5], ones(V, 1) ./ V, gma = 2, numepoch = 100 -> -7.076100e+06

function script
    %% load data
    %[x, vocab] = loadNIPSdata;
    [x, vocab] = loadNYTdata;
    %[x, vocab] = loadminidata;
    N = size(x, 2);
    V = length(vocab);
    assert(size(x, 1) == V, 'dimension of data and vocabulary is not agreed');
    
    %% set hyperparameters
    numepoch = 200;
    T = [1, 20, 10];
    L = length(T);
    assert(T(1) == 1, 'number of the root node should be one');
    children = zeros(1, L);
    children(1) = T(1);
    for l=2:L
        children(l) = children(l - 1) * T(l);
    end
    numnode = sum(children);
    
    alp = [2; 1; 0.5];
    bta = 1e-3 .* ones(V, 1);
    gma = 5;
    
    %% define paths
    [c, endstick] = treeidx(T, children);
    c = c';
    endidx = find(endstick == 1);
    
    %% set variational parameters
    v_prm = rand(2, numnode);
    v_prm(2, :) = gma;
    
    %c_prm = ncrp(gma, T, N);
    c_prm = rand(numnode, N);
    buf = c_prm(numnode - children(L) + 1:numnode, :);
    maxprob = max(buf, [], 1);
    buf = exp(buf - repmat(maxprob, children(L), 1));
    c_prm(numnode - children(L) + 1:numnode, :) = buf./repmat(sum(buf), children(L), 1);

    c_prm(2:numnode - children(L), :) = 0;
    i = numnode;
    c_i = c(i, :);
    L_i = L;
    while 2 < L_i
        c_i(L) = 0;
        [~, ~, ib] = intersect(c_i, c, 'rows');

        c_prm(ib, :) = c_prm(ib, :) + c_prm(i, :);

        i = i - 1;
        c_i = c(i, :);
        L_i = L - length(find(c_i == 0));
    end
    c_prm(1, :) = 1;
    
    phi_prm = repmat(bta, 1, numnode);
    
    theta_prm = repmat(alp, 1, N);
    
    theta = zeros(L, N);
    for n=1:N
        theta(:, n) = dirichletrnd(theta_prm(:, n));
    end
    
    z_prm = zeros(V, L, N);
    for k=1:V
        z_prm(k, :, :) = theta;
    end
    
    %% variational inference
    %ELBO = zeros(numepoch, 1);
    logv_mean = zeros(2, numnode);
    fullpathidx = sum(children(1:(L - 1))) + 1:numnode;
    endstick_flip = 1 - endstick;
    perplexity = zeros(N + 1, numepoch);
    numsamp = 100;
    y = zeros(numsamp, V);
    dist = zeros(numsamp, 1);
    I = sum(children(1:(L-1)));
    
    figure(1); plot(sum(x, 2)); title('histogram of words in the corpora'); drawnow;
    
    for epoch=1:numepoch
        fprintf('epoch %d [%s]\n', epoch, datetime);
        
        % update for variational parameters of v
        tic;
        fprintf(' update for variational parameters of v...');
        j = 1;
        for i=1:numnode
            if ~endstick(i)
                v_prm(1, i) = 1 + sum(c_prm(i, :));
                v_prm(2, i) = gma + sum(sum(c_prm(i+1:endidx(j), :)));
            else
                j = j + 1;
            end
        end
        t = toc;
        fprintf('completed (%3.3f sec)\n', t);
        
        tmp = psi(v_prm(1, :) + v_prm(2, :));
        logv_mean(1, :) = endstick_flip' .* (psi(v_prm(1, :)) - tmp);
        logv_mean(2, :) = endstick_flip' .* (psi(v_prm(2, :)) - tmp);
        
        %checkzerograd_v(gma, v_prm, endstick_flip, fullpathidx, c, c_prm, 1e-5, 5, 2);
        
        % update for variational parameters of theta
        tic;
        fprintf(' update for variational parameters of theta...');
        for n=1:N
            for l=1:L
                theta_prm(l, n) = alp(l) + sum(x(:, n) .* z_prm(:, l, n));
            end
        end
        t = toc;
        fprintf('completed (%3.3f sec)\n', t);
        
        logtheta_mean = psi(theta_prm) - repmat(psi(sum(theta_prm, 1)), L, 1);
        
        %checkzerograd_theta(z_prm, theta_prm, x, alp, 1e-6, 4, 2);
        
        % update for variational parameters of phi
        tic;
        fprintf(' update for variational parameters of phi...');
        for i=1:numnode
            buf = 0;
            l = L - length(find(c(i, :) == 0));

            for n=1:N
                buf = buf + c_prm(i, n) .* z_prm(:, l, n) .* x(:, n);
            end

            phi_prm(:, i) = bta + buf;
        end
        t = toc;
        fprintf('completed (%3.3f sec)\n', t);
        
        logphi_mean = psi(phi_prm) - repmat(psi(sum(phi_prm, 1)), V, 1);
        
        %checkzerograd_phi(z_prm, c_prm, phi_prm, x, c, bta, fullpathidx, 1e-5, 4, 2);
        
        % update for variational parameters of c
        tic;
        fprintf(' update for variational parameters of c (%d data)...', N);
        parfor n=1:N
            tmp = zeros(V, L);
            c_prmtmp = zeros(numnode, 1);
            
            % l = 1, root node
            buf_n = sum(z_prm(:, 1, n) .* x(:, n) .* logphi_mean(:, 1));
                
            for l=2:L
                tmp(:, l) = z_prm(:, l, n) .* x(:, n);
            end
            
            % the probability choosing root node (i = 1) is always 1 so it can be skipped
            for i=fullpathidx
                buf = buf_n;
                c_l = c(1, :);
                
                % L > l > 1
                for l=2:(L-1)
                    c_l(l) = c(i, l);
                    [~, ~, ib] = intersect(c_l, c, 'rows');

                    % expectation of p(c|v);
                    buf = buf + logv_mean(1, ib);
                    for j=1:(c(i, l) - 1)
                        buf = buf + logv_mean(2, ib - j);
                    end

                    % expectation of p(x|W, z)
                    buf = buf + sum(tmp(:, l) .* logphi_mean(:, ib));
                end

                % l = L, full path
                buf = buf + logv_mean(1, i);
                for j=1:(c(i, L) - 1)
                    buf = buf + logv_mean(2, i - j);
                end

                buf = buf + sum(tmp(:, L) .* logphi_mean(:, i));
                
                c_prmtmp(i) = buf - 1;
            end
            
            c_prm(:, n) = c_prmtmp;
        end
        c_prm(1, :) = 1;
        t = toc;
        fprintf('completed (%3.3f sec)\n', t);
        
        %checkzerograd_c(logv_mean, z_prm, exp(c_prm), logphi_mean, x, c, fullpathidx, 1e-9, 5, 2);

        % normalization
        buf = c_prm(numnode - children(L) + 1:numnode, :);
        maxprob = max(buf, [], 1);
        buf = exp(buf - repmat(maxprob, children(L), 1));
        c_prm(numnode - children(L) + 1:numnode, :) = buf./repmat(sum(buf), children(L), 1);
        
        c_prm(2:numnode - children(L), :) = 0;
        i = numnode;
        c_i = c(i, :);
        L_i = L;
        while 2 < L_i
            c_i(L) = 0;
            [~, ~, ib] = intersect(c_i, c, 'rows');
            
            c_prm(ib, :) = c_prm(ib, :) + c_prm(i, :);
            
            i = i - 1;
            c_i = c(i, :);
            L_i = L - length(find(c_i == 0));
        end

        %{
        % Pruning
        idx = find(sum(c_prm, 2) < 1e-6);
        
        if isempty(idx) == 0
            numprune = length(idx);
            fprintf('  pruning %d paths...', numprune);
            
            idxrmv = true(numnode, 1);
            idxrmv(idx) = false;
            
            [c_prm, phi_prm, v_prm, c, children, fullpathidx, endstick] = reduction(idxrmv, c_prm, phi_prm, v_prm, c, children);
            
            logphi_mean = psi(phi_prm) - repmat(psi(sum(phi_prm, 1)), V, 1);
            
            endidx = find(endstick == 1);
            endstick_flip = 1 - endstick;
            
            numnode = sum(children);
            
            logv_mean = zeros(2, numnode);
            tmp = psi(v_prm(1, :) + v_prm(2, :));
            logv_mean(1, :) = endstick_flip' .* (psi(v_prm(1, :)) - tmp);
            logv_mean(2, :) = endstick_flip' .* (psi(v_prm(2, :)) - tmp);
        end
            
        % Merging
        D = c_prm(fullpathidx, :) * c_prm(fullpathidx, :)';
        for i=fullpathidx
            for j=i+1:numnode
                D(i-I, j-I) = D(i-I, j-I) / (norm(c_prm(i, :)) * norm(c_prm(j, :)));
            end
        end
        D = triu(D, 1);
        [row, col] = find(D > 0.95);
        
        if isempty(row) == 0
            [~, idx] = unique(col);
            row = row(idx);
            col = col(idx);
            [~, idx] = unique(row);
            row = row(idx);
            col = col(idx);
            nummerge = length(row);
            
            fprintf('  merging %d paths...\n', nummerge);

            for k=1:nummerge
                i = I + row(k);
                j = I + col(k);

                c_prm(i, :) = c_prm(i, :) + c_prm(j, :);
            end

            idxrmv = true(numnode, 1);
            idxrmv(I + col) = false;
            
            [c_prm, phi_prm, v_prm, c, children, fullpathidx, endstick] = reduction(idxrmv, c_prm, phi_prm, v_prm, c, children);
            
            logphi_mean = psi(phi_prm) - repmat(psi(sum(phi_prm, 1)), V, 1);
            
            endidx = find(endstick == 1);
            endstick_flip = 1 - endstick;
            
            numnode = sum(children);
            
            logv_mean = zeros(2, numnode);
            tmp = psi(v_prm(1, :) + v_prm(2, :));
            logv_mean(1, :) = endstick_flip' .* (psi(v_prm(1, :)) - tmp);
            logv_mean(2, :) = endstick_flip' .* (psi(v_prm(2, :)) - tmp);
        end
        %}
        
        % update for variational parameters of z
        tic;
        fprintf(' update for variational parameters of z (%d data)...', N);
        z_prmcell = cell(N, 1);
        parfor n=1:N
            z_prmtmp = zeros(V, L);
            
            % l = 1, root node
            z_prmtmp(:, 1) = x(:, n) .* (logtheta_mean(1, n) + logphi_mean(:, 1)) - 1;
            
            % L > l > 1
            for l=2:(L-1)
                buf = 0;
                
                for i=fullpathidx
                    c_l = c(i, :);
                    c_l(l+1:L) = 0;
                    [~, ~, ib] = intersect(c_l, c, 'rows');

                    buf = buf + c_prm(i, n) .* logphi_mean(:, ib);
                end

                z_prmtmp(:, l) = x(:, n) .* (logtheta_mean(l, n) + buf) - 1;
            end
            
            % l = L, full path
            buf = 0;
            for i=fullpathidx
                buf = buf + c_prm(i, n) .* logphi_mean(:, i);
            end
            z_prmtmp(:, L) = x(:, n) .* (logtheta_mean(L, n) + buf) - 1;
            
            z_prmcell{n} = z_prmtmp;
        end
        
        parfor n=1:N
            z_prm(:, :, n) = z_prmcell{n};
        end
        
        t = toc;
        fprintf('completed (%3.3f sec)\n', t);

        %checkzerograd_z(logtheta_mean, exp(z_prm), c_prm, logphi_mean, x, c, fullpathidx, 1e-9, 4, 2);
        
        % normalization
        for n=1:N
            buf = z_prm(:, :, n);

            maxprob = max(buf, [], 2);
            buf = buf - repmat(maxprob, 1, L);
            buf = exp(buf);
            z_prm(:, :, n) = buf./repmat(sum(buf, 2), 1, L);
        end
        
        % Calculate perplexity
        tic;
        fprintf(' calculate log-likelihood...');
        phi_mean = phi_prm ./ repmat(sum(phi_prm), V, 1);
        theta_mean = theta_prm ./ repmat(sum(theta_prm), L, 1);
        
        parfor n=1:N
            buf = 0;
            
            for i=fullpathidx
                c_l = c(1, :);
                
                % l = 1, root node
                tmp = theta_mean(1, n) .* phi_mean(:, 1) .* x(:, n);
                
                % L > l > 1
                for l=2:(L-1)
                    c_l(l) = c(i, l);
                    [~, ~, ib] = intersect(c_l, c, 'rows');
                    
                    tmp = tmp + theta_mean(l, n) .* phi_mean(:, ib) .* x(:, n);
                end
                
                % l = L, full path
                tmp = tmp + theta_mean(L, n) .* phi_mean(:, i) .* x(:, n);
                
                buf = buf + c_prm(i, n) .* tmp;
            end
            
            hotidx = buf > 0;
            perplexity(n + 1, epoch) = sum(log(buf(hotidx)));
        end
        
        perplexity(1, epoch) = sum(perplexity(2:end, epoch));
        parfor n=1:N
            perplexity(n + 1, epoch) = exp(-perplexity(n + 1, epoch) / sum(x(:, n)));
        end
        perplexity(1, epoch) = exp(-perplexity(1, epoch) / sum(sum(x)));
        
        t = toc;
        fprintf('completed (%3.3f sec)\n', t);
        
        figure(2);
        subplot(2,1,1); image(perplexity(2:end, 1:epoch), 'CDataMapping', 'scaled'); title('perplexity per document');
        subplot(2,1,2); plot(perplexity(1, 1:epoch)); title(sprintf('perplexity (V = %d)', V));
        drawnow;
        
        fprintf(' perplexity out of the vocabulary consisting of %d words: %e\n', V, perplexity(1, epoch));
               
        % Recovery sampling
        y(:, :) = 0;
        n = randi(N);
        v_idx = find(x(:, n));
        figure(3);
        subplot(6, 1, 1); plot(x(:, n)); title(sprintf('n = %d (%d words)', n, sum(x(:,n))));
            
        parfor k=1:numsamp
            i = find(mnrnd(1, c_prm(fullpathidx, n))) + I;
            
            phi_mean = zeros(V, L);
            phi_mean(:, 1) = phi_prm(:, 1) ./ repmat(sum(phi_prm(:, 1)), V, 1);
            c_l = c(1, :);
            for l=2:(L-1)
                c_l(l) = c(i, l);
                [~, ~, ib] = intersect(c_l, c, 'rows');
                
                phi_mean(:, l) = phi_prm(:, ib) ./ repmat(sum(phi_prm(:, ib)), V, 1);
            end
            phi_mean(:, L) = phi_prm(:, i) ./ repmat(sum(phi_prm(:, i)), V, 1);
            
            for v=1:numel(v_idx)
                w = v_idx(v);
                
                for j=1:x(w, n)
                    l = mnrnd(1, z_prm(w, :, n)) == 1;
                    y(k, :) = y(k, :) + mnrnd(1, phi_mean(:, l));
                end
            end
            
            dist(k) = sum(abs(x(:, n) - y(k, :)'));
        end
        y_mean = mean(y);
        [~, imax] = max(dist);
        [~, imin] = min(dist);
        subplot(6, 1, 2); plot(y(imin, :)); title(sprintf('nearest (dist = %d)', dist(imin)/2));
        subplot(6, 1, 3); plot(y(imax, :)); title(sprintf('farthest (dist = %d)', dist(imax)/2));
        subplot(6, 1, 4); plot(y_mean); title('average');
        subplot(6, 1, 5); plot(y(1, :)); title(sprintf('sample 1 (dist = %d)', dist(1)/2));
        subplot(6, 1, 6); plot(y(2, :)); title(sprintf('sample 2 (dist = %d)', dist(2)/2));
        drawnow;
        
        % tree plot
        treevec = zeros(1, numnode);
        i = 2;
        c_i(:) = 0;
        for l=2:L
            for j=1:children(l)
                c_i(1:l-1) = c(i, 1:l-1);
                [~, ~, ib] = intersect(c_i, c, 'rows');

                treevec(i) = ib;
                i = i + 1;
            end
        end
        
        mainpath = sum(c_prm, 2) ./ N;
        idx = mainpath < 0;
        mainpath(idx) = eps;
        [a, b] = treelayout(treevec);
        figure(4); clf(4);
        figure(4); subplot(1,2,1); scatter(a, b, 300 .* mainpath); axis([0 1 0.1 0.9]); hold on;
        
        [~, idx] = sort(mainpath(fullpathidx), 'descend');
        idx = idx + I;
        
        numplot = 12;
        lalpha = mainpath(idx(1:numplot));
        textx = 1e-2 .* ones(L, 1);
        nodup = zeros(numplot, L) - 1;
            
        for i=1:numplot
            childidx = idx(i);
            parentidx = treevec(childidx);
            lcolor = [0.8 0.8 1-lalpha(i)];
            
            for l=L:-1:2
                if isempty(find(nodup == childidx, 1))
                    figure(4); subplot(1,2,1); plot([a(childidx) a(parentidx)], [b(childidx) b(parentidx)],...
                        'LineWidth', 0.3, 'Color', lcolor); hold on;

                    [~, widx] = sort(phi_prm(:, childidx), 'descend');
                    topNwrd = sprintf('%d (%2.1f%%)', childidx, 100*mainpath(childidx));
                    for j=1:8
                        topNwrd = strjoin({topNwrd, vocab{widx(j)}}, '\n');
                    end
                    figure(4); subplot(1,2,1); text(a(childidx), b(childidx) + 0.01, num2str(childidx)); hold on;
                    figure(4); subplot(1,2,2); text(textx(l), b(childidx), topNwrd);

                    textx(l) = textx(l) + 0.1;
                end
                
                nodup(i, l) = childidx;
                childidx = parentidx;
                parentidx = treevec(childidx);
            end
        end
        
        [~, widx] = sort(phi_prm(1, :), 'descend');
        topNwrd = sprintf('%d (%2.1f%%)', 1, 100*mainpath(1));
        for j=1:8
            topNwrd = strjoin({topNwrd, vocab{widx(j)}}, '\n');
        end
        figure(4); subplot(1,2,1); text(a(1), b(1) + 0.01, num2str(childidx)); hold off;
        figure(4); subplot(1,2,2); text(textx(1), b(1), topNwrd); axis([0 1 0.1 0.9]);
        drawnow;
    end
end

function [c_prm, phi_prm, v_prm, c, children, fullpathidx, endstick] = reduction(idxrmv, c_prm, phi_prm, v_prm, c, children)
    numnode = size(c, 1);
    L = size(c, 2);
    I = sum(children(1:L-1));
    numreduct = length(find(~idxrmv));
    
    c_prm = c_prm(idxrmv, :);
    phi_prm = phi_prm(:, idxrmv);
    v_prm = v_prm(:, idxrmv);
    c = c(idxrmv, :);

    numnode = numnode - numreduct;
    children(L) = children(L) - numreduct;
    children(2:(L-1)) = 1;

    for l=L:-1:3
        j = 1;

        for i=(I+1):(numnode-1)
            c(i, l - 1) = j;
            idxdif = c(i+1, l) - c(i, l);

            if idxdif > 1
                c(i+1, l) = c(i, l) + 1;
            elseif idxdif < 0
                j = j + 1;
                children(l - 1) = children(l - 1) + 1;

                if c(i+1, l) ~= 1
                    c(i+1, l) = 1;
                end
            end
        end
        
        c(numnode, l - 1) = j;
        
        idx = find(c(:, l - 1) > j);
        if isempty(idx) == 0
            c(idx, :) = [];
            c_prm(idx, :) = [];
            phi_prm(:, idx) = [];
            v_prm(:, idx) = [];
            numnode = numnode - length(idx);
        end
    end
    
    assert(numnode == sum(children), 'error in the merge operation');
    I = sum(children(1:(L-1)));
    fullpathidx = (I+1):numnode;

    for l=2:(L-1)
        j = 1;

        for i=sum(children(1:l-1))+1:sum(children(1:l))
            for ll=2:l
                c(i, ll) = j;
                j = j + 1;
            end
        end
    end

    endstick = zeros(numnode, 1);
    endstick(1) = 1;
    endstick(end) = 1;
    j = 2;
    for l=2:L
        for i=1:children(l)
            idxdif = c(j+1, l) - c(j, l);

            if idxdif < 0
                endstick(j) = 1;
            end

            j = j + 1;

            if j == numnode
                break;
            end
        end
    end
end

function checkzerograd_theta(z_prm, theta_prm, x, alp, eps, seed, num)
    prmname = 'theta_prm';
    rng(seed);
    
    ELBO = zeros(2, 1);
    h = [eps -eps];
    dim = size(theta_prm, 1);
    
    L = size(theta_prm, 1);
    N = size(theta_prm, 2);
    
    for numcount=1:num
        tic;
        dstidx = randi(N);
        prmidx = randi(dim);
        val = theta_prm(prmidx, dstidx);
        ELBO(:, 1) = 0;
    
        for loop=1:2
            theta_prm(prmidx, dstidx) = val + h(loop);
            logtheta_mean = psi(theta_prm) - repmat(psi(sum(theta_prm, 1)), L, 1);

            %p(theta|alpha)
            ELBO(loop) = ELBO(loop) + sum(sum(repmat((alp - 1), 1, N) .* logtheta_mean));

            %p(z|theta)
            for n=1:N
                for l=1:L
                    ELBO(loop) = ELBO(loop) + sum(x(:, n) .* z_prm(:, l, n)) .* logtheta_mean(l, n);
                end
            end

            %q(theta)
            buf = sum((theta_prm - 1) .* logtheta_mean) + gammaln(sum(theta_prm)) - sum(gammaln(theta_prm));
            ELBO(loop) = ELBO(loop) - sum(buf);
        end
        theta_prm(prmidx, dstidx) = val;
        t = toc;

        delta = (ELBO(1) - ELBO(2))/(2*h(1));
        fprintf('%e: numerical gradient of %s(%d, %d) = %e [%3.3f sec]\n', delta, prmname, prmidx, dstidx, val, t);
    end
end

function checkzerograd_phi(z_prm, c_prm, phi_prm, x, c, bta, fullpathidx, eps, seed, num)
    prmname = 'phi_prm';
    rng(seed);
    
    ELBO = zeros(2, 1);
    h = [eps -eps];
    dim = size(phi_prm, 1);
    
    L = size(c, 2);
    N = size(c_prm, 2);
    V = size(z_prm, 1);
    numnode = size(c_prm, 1);
    
    c_l = zeros(1, L);
    c_l(1) = 1;

    for numcount=1:num
        tic;
        dstidx = randi(numnode);
        prmidx = randi(dim);
        val = phi_prm(prmidx, dstidx);
        ELBO(:, 1) = 0;
    
        for loop=1:2
            phi_prm(prmidx, dstidx) = val + h(loop);
            logphi_mean = psi(phi_prm) - repmat(psi(sum(phi_prm, 1)), V, 1);

            %p(w|c,z,phi)
            for n=1:N
                for i=fullpathidx
                    buf = 0;
                    c_l(2:L) = 0;

                    buf = buf + sum(z_prm(:, 1, n) .* x(:, n) .* logphi_mean(:, 1));

                    for l=2:(L-1)
                        c_l(l) = c(i, l);
                        [~, ~, ib] = intersect(c_l, c, 'rows');

                        buf = buf + sum(z_prm(:, l, n) .* x(:, n) .* logphi_mean(:, ib));
                    end

                    buf = buf + sum(z_prm(:, L, n) .* x(:, n) .* logphi_mean(:, i));

                    ELBO(loop) = ELBO(loop) + c_prm(i, n) * buf;
                end
            end
        
            %p(phi|beta)
            ELBO(loop) = ELBO(loop) + sum(sum(repmat((bta - 1), 1, numnode) .* logphi_mean));

            %q(phi)
            buf = sum((phi_prm - 1) .* logphi_mean) + gammaln(sum(phi_prm)) - sum(gammaln(phi_prm));
            ELBO(loop) = ELBO(loop) - sum(buf);
        end
        phi_prm(prmidx, dstidx) = val;
        t = toc;

        delta = (ELBO(1) - ELBO(2))/(2*h(1));
        fprintf('%e: numerical gradient of %s(%d, %d) = %e [%3.3f sec]\n', delta, prmname, prmidx, dstidx, val, t);
    end
end

function checkzerograd_z(logtheta_mean, z_prm, c_prm, logphi_mean, x, c, fullpathidx, eps, seed, num)
    prmname = 'z_prm';
    rng(seed);
    
    ELBO = zeros(2, 1);
    h = [eps -eps];
    dim = size(z_prm, 1);
    
    L = size(c, 2);
    N = size(c_prm, 2);
    
    c_l = zeros(1, L);
    c_l(1) = 1;

    for numcount=1:num
        tic;
        prmidx = randi(dim);
        dstidx = randi(L);
        dtaidx = randi(N);
        val = z_prm(prmidx, dstidx, dtaidx);
        ELBO(:, 1) = 0;
    
        for loop=1:2
            z_prm(prmidx, dstidx, dtaidx) = val + h(loop);

            %p(z|theta)
            for n=1:N
                for l=1:L
                    ELBO(loop) = ELBO(loop) + sum(x(:, n) .* z_prm(:, l, n)) .* logtheta_mean(l, n);
                end
            end

            %p(w|c,z,phi)
            for n=1:N
                for i=fullpathidx
                    buf = 0;
                    c_l(2:L) = 0;

                    buf = buf + sum(z_prm(:, 1, n) .* x(:, n) .* logphi_mean(:, 1));

                    for l=2:(L-1)
                        c_l(l) = c(i, l);
                        [~, ~, ib] = intersect(c_l, c, 'rows');

                        buf = buf + sum(z_prm(:, l, n) .* x(:, n) .* logphi_mean(:, ib));
                    end

                    buf = buf + sum(z_prm(:, L, n) .* x(:, n) .* logphi_mean(:, i));

                    ELBO(loop) = ELBO(loop) + c_prm(i, n) * buf;
                end
            end
        
            %q(z)
            ELBO(loop) = ELBO(loop) - sum(sum(sum(z_prm .* log(z_prm))));
        end
        z_prm(prmidx, dstidx, dtaidx) = val;
        t = toc;

        delta = (ELBO(1) - ELBO(2))/(2*h(1));
        fprintf('%e: numerical gradient of %s(%d, %d, %d) = %e [%3.3f sec]\n', delta, prmname, prmidx, dstidx, dtaidx, val, t);
    end
end

function checkzerograd_c(logv_mean, z_prm, c_prm, logphi_mean, x, c, fullpathidx, eps, seed, num)
    prmname = 'c_prm';
    rng(seed);
    
    ELBO = zeros(2, 1);
    h = [eps -eps];
    dim = size(c_prm, 1);
    
    L = size(c, 2);
    N = size(c_prm, 2);
    
    c_l = zeros(1, L);
    c_l(1) = 1;

    for numcount=1:num
        tic;
        prmidx = randi(dim);
        dstidx = randi(N);
        val = c_prm(prmidx, dstidx);
        ELBO(:, 1) = 0;
    
        for loop=1:2
            c_prm(prmidx, dstidx) = val + h(loop);

            %p(c|v)
            for n=1:N
                for i=fullpathidx
                    buf = 0;
                    c_i = c(i, :);
                    c_l(2:L) = 0;

                    for l=2:(L-1)
                        c_l(l) = c_i(l);
                        [~, ~, ib] = intersect(c_l, c, 'rows');

                        buf = buf + logv_mean(1, ib);
                        for j=1:(c_i(l) - 1)
                            buf = buf + logv_mean(2, ib - j);
                        end
                    end

                    buf = buf + logv_mean(1, i);
                    for j=1:(c_i(L) - 1)
                        buf = buf + logv_mean(2, i - j);
                    end

                    ELBO(loop) = ELBO(loop) + c_prm(i, n) * buf;
                end
            end

            %p(w|c,z,phi)
            for n=1:N
                for i=fullpathidx
                    buf = 0;
                    c_l(2:L) = 0;

                    buf = buf + sum(z_prm(:, 1, n) .* x(:, n) .* logphi_mean(:, 1));

                    for l=2:(L-1)
                        c_l(l) = c(i, l);
                        [~, ~, ib] = intersect(c_l, c, 'rows');

                        buf = buf + sum(z_prm(:, l, n) .* x(:, n) .* logphi_mean(:, ib));
                    end

                    buf = buf + sum(z_prm(:, L, n) .* x(:, n) .* logphi_mean(:, i));

                    ELBO(loop) = ELBO(loop) + c_prm(i, n) * buf;
                end
            end
        
            %q(c)
            ELBO(loop) = ELBO(loop) - sum(sum(c_prm(fullpathidx, :).*log(c_prm(fullpathidx, :))));
        end
        c_prm(prmidx, dstidx) = val;
        t = toc;

        delta = (ELBO(1) - ELBO(2))/(2*h(1));
        fprintf('%e: numerical gradient of %s(%d, %d) = %e, [%3.3f sec]\n', delta, prmname, prmidx, dstidx, val, t);
    end
end

function checkzerograd_v(gma, v_prm, endstick_flip, fullpathidx, c, c_prm, eps, seed, num)
    prmname = 'v_prm';
    rng(seed);
    
    ELBO = zeros(2, 1);
    h = [eps -eps];
    dim = size(v_prm, 1);
    
    L = size(c, 2);
    N = size(c_prm, 2);
    
    c_l = zeros(1, L);
    c_l(1) = 1;
    
    numnode = size(c_prm, 1);
    logv_mean = zeros(2, numnode);
    
    for numcount=1:num
        tic;
        dstidx = randi(numnode);
        prmidx = randi(dim);
        val = v_prm(prmidx, dstidx);
        ELBO(:, 1) = 0;
    
        for loop=1:2
            v_prm(prmidx, dstidx) = val + h(loop);

            % get expectation
            tmp = psi(v_prm(1, :) + v_prm(2, :));
            logv_mean(1, :) = endstick_flip' .* (psi(v_prm(1, :)) - tmp);
            logv_mean(2, :) = endstick_flip' .* (psi(v_prm(2, :)) - tmp);

            %p(v|lambda)
            ELBO(loop) = ELBO(loop) + (gma - 1) * sum(logv_mean(2, :));

            %p(c|v)
            for n=1:N
                for i=fullpathidx
                    buf = 0;
                    c_i = c(i, :);
                    c_l(2:L) = 0;

                    for l=2:(L-1)
                        c_l(l) = c_i(l);
                        [~, ~, ib] = intersect(c_l, c, 'rows');

                        buf = buf + logv_mean(1, ib);
                        for j=1:(c_i(l) - 1)
                            buf = buf + logv_mean(2, ib - j);
                        end
                    end

                    buf = buf + logv_mean(1, i);
                    for j=1:(c_i(L) - 1)
                        buf = buf + logv_mean(2, i - j);
                    end

                    ELBO(loop) = ELBO(loop) + c_prm(i, n) * buf;
                end
            end

            %q(v)
            buf = gammaln(v_prm(1, :) + v_prm(2, :)) - gammaln(v_prm(1, :)) - gammaln(v_prm(2, :));
            buf = buf + (v_prm(1, :) - 1).*logv_mean(1, :) + (v_prm(2, :) - 1).*logv_mean(2, :);
            ELBO(loop) = ELBO(loop) - sum(buf);
        end
        v_prm(prmidx, dstidx) = val;
        t = toc;

        delta = (ELBO(1) - ELBO(2))/(2*h(1));
        fprintf('%e: numerical gradient of %s(%d, %d) = %e [%3.3f sec]\n', delta, prmname, prmidx, dstidx, val, t);
    end
end

function [c, endstick] = treeidx(T, children)
    L = length(T);
    numnode = sum(children);
    
    endstick = zeros(numnode, 1);
    endstick(1) = 1;
    c = zeros(L, numnode);
    i = 1;
    c(1, i) = 1;
    for l=2:L
        pathidx = ones(l, 1);
        
        for j=(i+1):(i+children(l))
            c(1:l, j) = pathidx;
            
            [pathidx, isend] = nextidx(pathidx, l, T);
            endstick(j) = isend;
        end
        
        i = i + children(l);
    end
end

function [pathidx, isend] = nextidx(pathidx, l, T)
    pathidx(l, 1) = pathidx(l, 1) + 1;
    isend = 0;
    
    if l > 1 && pathidx(l, 1) > T(l)
        pathidx(l, 1) = 1;
        isend = 1;

        pathidx = nextidx(pathidx, l-1, T);
    end
end