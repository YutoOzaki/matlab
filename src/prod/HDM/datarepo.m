classdef datarepo
    properties
        vocabulary
        K, M, J, V
        n_j, n_k, n_kv, n_jt, n_jtv, n_jk
        m_k
        w, topicMat, tableMat, phi
    end
    
    methods
        function obj = datarepo(w, J, V, vocabulary)
            assert(length(w) == J, 'number of documents and actual data (type: cell) is inconsistent');
            assert(length(vocabulary) == V, 'number of vocabularies and actual data (type: cell) is inconsistent');
            
            obj.vocabulary = vocabulary;
            obj.w = w;
            obj.J = J;
            obj.V = V;
            
            obj = obj.init();
        end
        
        function obj = init(obj)
            obj.topicMat = obj.w;
            obj.tableMat = obj.w;
            
            obj.K = 0;
            obj.M = 0;
            
            obj.n_k = zeros(obj.K, 1);
            obj.n_kv = zeros(obj.K, obj.V);
            
            obj.n_j = obj.count_j(obj.w);

            obj.n_jt = cell(obj.J, 1);
            obj.n_jtv = cell(obj.J, 1);
            obj.phi = cell(obj.J, 1);
            for j=1:obj.J
                obj.n_jt{j} = [];
                obj.n_jtv{j} = [];
                obj.phi{j} = [];
            end
            
            obj.m_k = zeros(obj.K, 1);
        end
        
        function n_j = count_j(obj, w)
            n_j = cell2mat(cellfun(@(x) length(x), w, 'UniformOutput', false));
        end
        
        %% count n_jk
        function n_jk = count_jk(obj)
            topicMat = obj.topicMat;
            J = obj.J;
            K = obj.K;
            
            n_jk = zeros(K, J);

            for k=1:K
                n_jk(k,:) = cell2mat(cellfun(@(x) length(find(x==k)), topicMat, 'UniformOutput', false))';
            end
            
            assert(isequal(sum(n_jk, 1), obj.n_j'), 'n_jk and n_j is inconsistent');
        end
    end
end