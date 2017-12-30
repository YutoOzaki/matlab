%nCRP-LDA with variational inference
function script
    %% load data
    nyt_data = 'C:\Users\yuto\Documents\MATLAB\data\topicmodel\NewYorkTimesNews\nyt_data.txt';
    nyt_vocab = 'C:\Users\yuto\Documents\MATLAB\data\topicmodel\NewYorkTimesNews\nyt_vocab.dat';
    [x, vocab] = loaddata(nyt_data, nyt_vocab);
    %[x, vocab] = minidata;
    N = size(x, 2);
    V = length(vocab);
    assert(size(x, 1) == V, 'dimension of data and vocabulary is not agreed');
    
    %% set hyperparameters
    numepoch = 100;
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
    bta = ones(V, 1) ./ V;
    gma = 2;
    
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
    c_l = zeros(1, L);
    c_l(1) = 1;
    logv_mean = zeros(2, numnode);
    fullpathidx = sum(children(1:(L - 1))) + 1:numnode;
    endstick_flip = 1 - endstick;
    perplexity = zeros(N + 1, numepoch);
    numsamp = 100;
    y = zeros(numsamp, V);
    dist = zeros(numsamp, 1);
    if V > 9
        numwrd = 10;
    else
        numwrd = V;
    end
    
    figure(1); plot(sum(x, 2)); title('histogram of words in the corpora');
    
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
        
        %checkzerograd_v(gma, v_prm, endstick_flip, fullpathidx, c, c_prm, 1e-6, 5, 2);
        
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
        i = 1;
        for l=1:L
            for j=1:children(l)
                buf = 0;
                
                for n=1:N
                    buf = buf + c_prm(i, n) .* z_prm(:, l, n) .* x(:, n);
                end
                
                phi_prm(:, i) = bta + buf;
                i = i + 1;
            end
        end
        t = toc;
        fprintf('completed (%3.3f sec)\n', t);
        
        logphi_mean = psi(phi_prm) - repmat(psi(sum(phi_prm, 1)), V, 1);
        
        %checkzerograd_phi(z_prm, c_prm, phi_prm, x, c, bta, fullpathidx, 1e-6, 4, 2);
        
        % update for variational parameters of c
        tic;
        fprintf(' update for variational parameters of c (%d data)...', N);
        for n=1:N
            fprintf('%d', n);
            
            % the probability choosing root node (i = 1) is always 1 so it can be skipped
            for i=fullpathidx
                c_l(2:L) = 0;
                
                % l = 1, root node
                buf = sum(z_prm(:, 1, n) .* x(:, n) .* logphi_mean(:, 1));
                
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
                    buf = buf + sum(z_prm(:, l, n) .* x(:, n) .* logphi_mean(:, ib));
                end

                % l = L, full path
                buf = buf + logv_mean(1, i);
                for j=1:(c(i, L) - 1)
                    buf = buf + logv_mean(2, i - j);
                end

                buf = buf + sum(z_prm(:, L, n) .* x(:, n) .* logphi_mean(:, i));
                
                c_prm(i, n) = buf - 1;
            end

            fprintf(repmat('\b', [1, length(num2str(n))]));
        end
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

        % update for variational parameters of z
        tic;
        fprintf(' update for variational parameters of z (%d data)...', N);
        for n=1:N
            fprintf('%d', n);
            c_l(2:L) = 0;
            
            % l = 1, root node
            z_prm(:, 1, n) = x(:, n) .* (logtheta_mean(1, n) + logphi_mean(:, 1)) - 1;
            
            % L > l > 1
            for l=2:(L-1)
                buf = 0;
                
                for i=fullpathidx
                    c_l(l) = c(i, l);
                    [~, ~, ib] = intersect(c_l, c, 'rows');

                    buf = buf + c_prm(i, n) .* logphi_mean(:, ib);
                end

                z_prm(:, l, n) = x(:, n) .* (logtheta_mean(l, n) + buf) - 1;
            end
            
            % l = L, full path
            buf = 0;
            for i=fullpathidx
                buf = buf + c_prm(i, n) .* logphi_mean(:, i);
            end
            z_prm(:, L, n) = x(:, n) .* (buf + logtheta_mean(L, n)) - 1;
            
            fprintf(repmat('\b', [1, length(num2str(n))]));
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

        %{
        % Calculate ELBO
        fprintf(' check ELBO...');
        tic;
        
        %p(v|lambda)
        ELBO(epoch) = ELBO(epoch) + (gma - 1) * sum(logv_mean(2, :));
        
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
                    
                ELBO(epoch) = ELBO(epoch) + c_prm(i, n) * buf;
            end
        end
        
        %q(v)
        buf = gammaln(v_prm(1, :) + v_prm(2, :)) - gammaln(v_prm(1, :)) - gammaln(v_prm(2, :));
        buf = buf + (v_prm(1, :) - 1).*logv_mean(1, :) + (v_prm(2, :) - 1).*logv_mean(2, :);
        ELBO(epoch) = ELBO(epoch) - sum(buf);
        
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
                
                ELBO(epoch) = ELBO(epoch) + c_prm(i, n) * buf;
            end
        end
        
        %q(c)
        ELBO(epoch) = ELBO(epoch) - sum(sum(c_prm(fullpathidx, :).*log(c_prm(fullpathidx, :))));
        
        %q(z)
        ELBO(epoch) = ELBO(epoch) - sum(sum(sum(z_prm .* log(z_prm))));
        
        %p(phi|beta)
        ELBO(epoch) = ELBO(epoch) + sum(sum(repmat((bta - 1), 1, numnode) .* logphi_mean));
        
        %q(phi)
        buf = sum((phi_prm - 1) .* logphi_mean) + gammaln(sum(phi_prm)) - sum(gammaln(phi_prm));
        ELBO(epoch) = ELBO(epoch) - sum(buf);
        
        %p(theta|alpha)
        ELBO(epoch) = ELBO(epoch) + sum(sum(repmat((alp - 1), 1, N) .* logtheta_mean));
        
        %p(z|theta)
        for n=1:N
            for l=1:L
                ELBO(epoch) = ELBO(epoch) + sum(x(:, n) .* z_prm(:, l, n)) .* logtheta_mean(l, n);
            end
        end
        
        %q(theta)
        buf = sum((theta_prm - 1) .* logtheta_mean) + gammaln(sum(theta_prm)) - sum(gammaln(theta_prm));
        ELBO(epoch) = ELBO(epoch) - sum(buf);
        
        t = toc;
        fprintf('%3.3f (%3.3f sec)\n', ELBO(epoch), t);
        figure(1); plot(ELBO); xlim([0 epoch]); drawnow;
        %}
        
        % Calculate perplexity
        tic;
        fprintf(' calculate log-likelihood...');
        phi_mean = phi_prm ./ repmat(sum(phi_prm), V, 1);
        theta_mean = theta_prm ./ repmat(sum(theta_prm), L, 1);
        
        for n=1:N
            fprintf('%d', n);
            buf = 0;
            
            for i=fullpathidx
                c_l(2:L) = 0;
                
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
            fprintf(repmat('\b', [1, length(num2str(n))]));
        end
        
        perplexity(1, epoch) = sum(perplexity(2:end, epoch));
        t = toc;
        fprintf('completed (%3.3f sec)\n', t);
        
        figure(2);
        subplot(2,1,1); image(perplexity(2:end, 1:epoch), 'CDataMapping', 'scaled');
        subplot(2,1,2); plot(perplexity(1, 1:epoch));
        drawnow;
        
        fprintf(' log-likelihood: %e\n', perplexity(1, epoch));
        
        % Growing, pruning and merging of branches
        
        % Recovery sampling
        y(:, :) = 0;
        n = randi(N);
        v_idx = find(x(:, n));
        I = sum(children(1:(L-1)));
        figure(3);
        subplot(6, 1, 1); plot(x(:, n)); title(sprintf('n = %d', n));
            
        for k=1:numsamp
            for v=1:length(v_idx)
                w = v_idx(v);
                
                for j=1:x(w, n)
                    c_i(:) = 0;
                    
                    l = find(mnrnd(1, z_prm(w, :, n)));
                    i = find(mnrnd(1, c_prm(fullpathidx, n)));
                    c_i(1:l) = c(I + i, 1:l);
                    [~, ~, ib] = intersect(c_i, c, 'rows');
                    phi = dirichletrnd(phi_prm(:, ib));
                    y(k, :) = y(k, :) + mnrnd(1, phi);
                end
            end
            
            dist(k) = sum(abs(x(:, n) - y(k, :)'));
        end
        y_mean = mean(y);
        [~, i] = min(dist);
        subplot(6, 1, 2); plot(y(i, :)); title('nearest');
        subplot(6, 1, 3); plot(y_mean); title('average');
        subplot(6, 1, 4); plot(y(1, :)); title('sample 1');
        subplot(6, 1, 5); plot(y(2, :)); title('sample 2');
        subplot(6, 1, 6); plot(y(3, :)); title('sample 3');
        drawnow;
        
        % Word topic visualization
        numpath = 1;
        wrdbox = cell(L*numpath, 2 + numwrd);
        for k=1:10
            n = randi(N);
            p = c_prm((numnode - children(L) + 1):numnode, n);
            [~, I_c] = sort(p, 'descend');
            [~, I_theta] = sort(theta_prm(:, n), 'descend');
            fprintf('[n = %d] ', n);
            
            for j=1:numpath
                i = numnode - children(L) + I_c(j);
                c_j = c(i, :);

                for l=1:L
                    c_l(1:L) = 0;
                    c_l(1:I_theta(l)) = c_j(1:I_theta(l));
                    [~, ~, ib] = intersect(c_l, c, 'rows');

                    [~, I_phi] = sort(phi_prm(:, ib), 'descend');
                    wrdbox((j-1)*L + l, 3:2+numwrd) = vocab(I_phi(1:numwrd))';
                    wrdbox((j-1)*L + l, 1) = num2cell(I_theta(l));
                    wrdbox((j-1)*L + l, 2) = num2cell(ib);
                end

                fprintf(' q(c_%d) = %3.3f, ', i, c_prm(i, n));
            end
            
            fprintf('\n');
            disp(wrdbox);
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

function [x, vocab] = minidata
    vocab = {'a','b','c'};
    x = [0 2 0 0 0 2;0 1 0 2 2 0;1 0 1 0 0 2];
end

function [x, vocab] = loaddata(datapath, vocabpath)
    fileID = fopen(datapath, 'r');
    data = textscan(fileID, '%s');
    fclose(fileID);
    data = data{1};
    
    fileID = fopen(vocabpath, 'r');
    vocab = textscan(fileID, '%s');
    fclose(fileID);
    vocab = vocab{1};
    
    N = length(data);
    V = length(vocab);
    x = zeros(V, N);
    fprintf('formatting data (%d documents)...', N);
    
    for n=1:N
        fprintf('%d', n);
        
        str = strsplit(data{n}, ',');
        A = cell2mat(cellfun(@(s) str2double(strsplit(s, ':'))', str, 'UniformOutput', false))';
        
        assert(size(A, 1) == length(unique(A(:, 1))), 'bag of words with duplicated elements!');
        
        x(A(:, 1), n) = A(:, 2);
        
        fprintf(repmat('\b', [1, length(num2str(n))]));
    end
    
    fprintf('completed\n');
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