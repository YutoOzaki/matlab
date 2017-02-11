%% deletion of topics
function [n_k, n_kv, m_k, topicMat, phi] = deleteTopics(k, K, V, J, n_k, n_kv, m_k, topicMat, phi)
    n_k_new = zeros(K, 1);
    n_k_new(1:k-1) = n_k(1:k-1);
    n_k_new(k:K) = n_k(k+1:K+1);
    n_k = n_k_new;

    n_kv_new = zeros(K, V);
    n_kv_new(1:k-1,:) = n_kv(1:k-1,:);
    n_kv_new(k:K,:) = n_kv(k+1:K+1,:);
    n_kv = n_kv_new;

    m_k_new = zeros(K, 1);
    m_k_new(1:k-1) = m_k(1:k-1);
    m_k_new(k:K) = m_k(k+1:K+1);
    m_k = m_k_new;

    idx_k = cellfun(@(x) find(x > k), topicMat, 'UniformOutput', false);
    idx_p = cellfun(@(x) find(x > k), phi, 'UniformOutput', false);
    for i=1:J
        topicMat{i}(idx_k{i}) = topicMat{i}(idx_k{i}) - 1;
        phi{i}(idx_p{i}) = phi{i}(idx_p{i}) - 1;
    end
end