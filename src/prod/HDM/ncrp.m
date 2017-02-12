classdef ncrp < handle
    properties
        alpha
        J
        count_n
        child
    end
    
    methods
        function obj = crptree(eta, J)
            obj.alpha = eta(1);
            obj.J = J;
            
            [~, count_n] = crp(obj.alpha, obj.J);
            obj.count_n = count_n(1:end-1);
            
            %{
            fprintf(repmat(' ', 1, length(eta)));
            fprintf('%d -> [', obj.J);
            for k=1:length(obj.count_n)
                fprintf(' %d', obj.count_n(k));
            end
            fprintf(' ] (%3.3f) \n', obj.alpha);
            %}
            
            if length(eta) > 1
                eta = eta(2:end);
                
                K = length(obj.count_n);
                obj.child = cell(K, 1);
                
                for k=1:K
                    obj.child{k} = crptree(eta, obj.count_n(k));
                end
            end
        end
    end
end