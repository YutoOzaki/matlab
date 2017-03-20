function [pi, z] = categoryassignment(eta, alpha, pi, z)
    z = zcomp(z);
    N = size(z, 1);
    K = size(pi{1}, 1);
    L = length(alpha);
    
    %disp(sortrows(z, 4));%
    
    for l=L:-1:3
        J = size(pi{l}, 2);
        
        G = zeros(l - 1, 1);
        for ll=(l - 1):-1:1
            G(ll) = size(pi{ll}, 2);
        end
        
        n_g = zeros(G(end) + 1, 1);
        for g=1:G(end)
            idx = find(z(:, l - 1) == g);
            n_g(g) = length(idx);
        end
        n_g(G(end) + 1) = eta(l - 2);
        
        logp = zeros(sum(G), 1);
        pibag = cell(sum(G(1:end - 1)), 1);
        
        for j=1:J
            idx_z = find(z(:, l) == j);
            g = z(idx_z, l - 1);
            assert(length(unique(g)) == 1, 'z is corrupted');
            g = g(1);
            n_g(g) = n_g(g) - length(idx_z);
            
            z(idx_z, l - 1) = -1;
            
            if n_g(g) == 0
                assert(isempty(find(z(:, l - 1) == g, 1)), 'z is corrupted');
                n_g = utils.deleteElement(n_g, g);
                idx = g;
                
                for ll=(l - 1):-1:2
                    if length(idx) == 1
                        pi_g = pi{ll};
                        pi_g = utils.deleteElement(pi_g', g);
                        pi{ll} = pi_g';
                        G(ll) = G(ll) - 1;
                        idx = z(:, ll) > g;
                        z(idx, ll) = z(idx, ll) - 1;
                        
                        z(idx_z, ll) = -1;
                    else
                        break;
                    end
                    
                    g = z(idx_z, ll - 1);
                    assert(length(unique(g)) == 1, 'z is corrupted');
                    g = g(1);
                    idx = find(z(:, ll - 1) == g);
                end
                
                logp = zeros(sum(G), 1);
            end
            
            %disp(sortrows(z, 4));%
            
            theta = pi{l}(:, j);
            idx = theta == 0;
            theta(idx) = eps;
            logtheta = log(theta);
            
            for g=1:G(end)
                beta_g = pi{l - 1}(:, g);
                idx = beta_g == 0;
                beta_g(idx) = eps;
                
                a = alpha(l) .* beta_g;
                
                idx = find(z(:, l - 1) == g);
                idx = find(z(:, l - 2) == z(idx(1), l - 2));
                n = length(idx);
                if ~isempty(find(z(idx, l - 1) == -1, 1))
                    n = n - length(idx_z);
                end
                denm = n + eta(l - 2);
                crplik = log(n_g(g)/denm);
                %fprintf('crp likeliood: %d / (%d + %3.3f) <g = %g>\n', n_g(g), n, eta(l - 2), g);
                
                logp(g) = loglikhelper(logtheta, a, crplik);
            end
            
            count = G(end) + 1;
            for ll=(length(G) - 1):-1:1
                pibag_g = zeros(K, length(G) - ll);
                
                for g=1:G(ll)
                    beta_g = pi{ll}(:, g);
                    for lll=(ll + 1):(l - 1)
                        beta_g = hsbp(alpha(lll), beta_g);
                        pibag_g(:, l - lll) = beta_g;
                    end
                    idx = beta_g == 0;
                    beta_g(idx) = eps;

                    a = alpha(l) .* beta_g;
                    
                    idx = find(z(:, ll) == g);
                    n = length(idx);
                    if ~isempty(find(z(idx, l - 1) == -1, 1))
                        n = n - length(idx_z);
                    end
                    denm = n + eta(ll);
                    crplik = log(eta(ll)/denm);
                    %fprintf('crp likeliood: %3.3f / (%d + %3.3f) <g = %g>\n', eta(ll), n, eta(ll), g);

                    logp(count) = loglikhelper(logtheta, a, crplik);
                    
                    pibag{count - G(end)} = pibag_g;
                    count = count + 1;
                end
            end
            
            parentidx = find(logmnrnd(logp));
            
            pibagidx = parentidx - G(end);
            if pibagidx > 0
                pi_new = pibag{pibagidx};
                count = 1;
            end
            
            for ll=(l - 1):-1:2
                if parentidx <= G(ll)
                    newparent = parentidx;
                    
                    nextparentidx = z(:, ll) == newparent;
                    nextparent = z(nextparentidx, ll - 1);
                else
                    if ll == (l - 1)
                        n_g = utils.insertElement(n_g, 0, G(ll) + 1);
                    end
                    nextparent = parentidx - G(ll);
                    
                    pi{ll} = [pi{ll} pi_new(:, count)];
                    count = count + 1;
                    
                    G(ll) = G(ll) + 1;
                    logp = utils.insertElement(logp, 0, 1);
                    pibag = cell(sum(G(1:end - 1)), 1);
                    
                    newparent = G(ll);
                end
                
                assert(length(unique(nextparent)) == 1, 'mixed parent');
                
                z(idx_z, ll) = newparent;
                parentidx = nextparent(1);
                
                zset = unique(z(:, ll));
                diffidx = setdiff(1:G(ll), zset);
                if ~isempty(diffidx)
                    G(ll) = G(ll) - 1;
                    
                    pi_g = pi{ll};
                    pi_g = utils.deleteElement(pi_g', diffidx);
                    pi{ll} = pi_g';
                    
                    idx = z(:, ll) > diffidx;
                    z(idx, ll) = z(idx, ll) - 1;
                    
                    logp = utils.deleteElement(logp, 1);
                    pibag = cell(sum(G(1:end - 1)), 1);
                end
            end
            
            g = z(idx_z, l - 1);
            assert(length(unique(g)) == 1, 'z is corrupted');
            g = g(1);
            n_g(g) = n_g(g) + length(idx_z);
            assert(sum(n_g(1:end - 1)) == N, 'n_g and N is inconsistent');
            
            assert(length(logp) == sum(G), 'logp and G is inconsistent');
            pisize = cell2mat(cellfun(@(x) size(x, 2), pi, 'UniformOutput', false));
            assert(isequal(G, pisize(1:l - 1)), 'pi and G is inconsistent');
            zmax = max(z, [], 1);
            assert(isequal(zmax(:), pisize), 'pi and z is inconsistent');
            for i=1:L
                assert(length(unique(z(:, i))) == zmax(i), 'z is corrupted');
            end
        end
    end
    
    z = z(:, 2:end - 1);
end

function z = zcomp(z)
    N = size(z, 1);
    z = [ones(N, 1) z (1:N)'];
end

function loglik = loglikhelper(logtheta, a, crplik)
    logbeta = sum(gammaln(a)) - gammaln(sum(a));
    loglik = sum((a - 1) .* logtheta) - logbeta + crplik;
end