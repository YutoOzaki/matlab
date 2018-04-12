classdef reparamtrans < basenode
    properties
        input, prms, grad, optm
        eps, J, L
    end
    
    methods
        function obj = reparamtrans(J, L)
            obj.prms = struct();
            obj.J = J;
            obj.L = L;
        end
        
        function output = forwardprop(obj, input)
            obj.input = input;
            
            [~, batchsize] = size(input.mu);
            output = zeros(obj.J, batchsize, obj.L);
            for l=1:(obj.L)
                output(:, :, l) = input.mu + bsxfun(@times, sqrt(input.sig), obj.eps(:, l));
            end
        end
        
        function delta = backwardprop(obj, input)
            dzdmu = sum(input, 3);
            
            dzdsig = 0;
            for l=1:obj.L
                A = bsxfun(@times, input(:, :, l), obj.eps(:, l));
                
                dzdsig = dzdsig + 0.5.*A./sqrt(obj.input.sig);
            end
            
            delta = struct(...
                'mu', dzdmu,...
                'sig', dzdsig...
                );
        end
        
        function init(obj)
            obj.eps = mvnrnd(zeros(1, obj.J), diag(ones(obj.J, 1)), obj.L)';
        end
        
        function update(obj)
        end
    end
end