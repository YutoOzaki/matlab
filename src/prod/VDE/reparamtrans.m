classdef reparamtrans < basenode
    properties
        input, prms, grad
        eps, J, L
    end
    
    methods
        function obj = reparamtrans(J, L)
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
        end
        
        function init(obj)
            obj.eps = mvnrnd(zeros(1, obj.J), diag(ones(obj.J, 1)), obj.L)';
        end
        
        function update(obj)
        end
    end
end