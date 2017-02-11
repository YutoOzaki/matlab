function hdp_testdata_result(K, gam, alpha, beta, pi, theta, phi, groundtruth)
    fprintf('K     = %d (%d, E(M) = %3.3f)\n', K, groundtruth.K, groundtruth.M_mean);
    fprintf('gamma = %3.3f (%3.3f)\n', gam, groundtruth.gamma);
    fprintf('alpha = %3.3f (%3.3f)\n', alpha, groundtruth.alpha);
    fprintf('beta  = %3.3f (%3.3f)\n', beta, groundtruth.beta);
    
    rank = zeros(groundtruth.K, 1);
    idx_hist = zeros(K+1, 1);
    for k=1:K
        for kk=1:groundtruth.K
            acor = xcorr(theta(1+k,:), groundtruth.theta(kk,:));
            [~,I] = max(abs(acor));
            rank(kk) = abs(acor(I));
        end
        
        [~, rank_idx] = sort(rank, 'descend');
        counter = 1;
        idx = rank_idx(counter);
        
        if k <= groundtruth.K
            while ~isempty(find(idx_hist == idx, 1))
                counter = counter + 1;
                idx = rank_idx(counter);
            end
        else
            idx = k;
        end
        
        idx_hist(1+k) = idx;
    end
    
    if K < groundtruth.K
        df = setdiff(1:groundtruth.K, idx_hist);
        
        for i=length(df):-1:1;
            idx = idx_hist > df(i);
            idx_hist(idx) = idx_hist(idx) - 1;
        end
    end
    
    idx_hist(1) = K + 1;
    
    pi_buf = pi;
    theta_buf = theta;
    for k=1:K+1
        pi_buf(idx_hist(k)) = pi(k);
        theta_buf(idx_hist(k),:) = theta(k,:);
    end
    
    figure(6); 
        subplot(2,12,1); imagesc(groundtruth.pi); caxis([0 1]); set(gca, 'XTick', []); title('global-level (true)');
        subplot(2,12,3:12); imagesc(groundtruth.theta); caxis([0 1]); title('local-level (true)');
        subplot(2,12,13); imagesc(pi_buf); caxis([0 1]); set(gca, 'XTick', []); title('global-level');
        subplot(2,12,15:24); imagesc(theta_buf); caxis([0 1]); title('local-level');
        
    figure(7);
        subplot(2,1,1); imagesc(groundtruth.H'); caxis([0 1]); title('topic-word distribution (true)');
        subplot(2,1,2); imagesc(phi(:,idx_hist(2:K+1))'); caxis([0 1]); title('topic-word distribution');
    
    figure(8);
        subplot(3,1,1); plot(groundtruth.pi); hold on;
        plot(pi_buf, '-.m'); hold off;
        subplot(3,1,2); plot(groundtruth.theta(:,1)); hold on;
        plot(theta_buf(:,1), '-.m'); hold off;
        subplot(3,1,3); plot(groundtruth.theta(:,2)); hold on;
        plot(theta_buf(:,2), '-.m'); hold off;
end