classdef reparamtrans < basenode
    properties
        input, prms, delta
    end
    
    methods
        function obj = reparamtrans(J, L)
            obj.prms = struct(...
                'eps', [],...
                'J', J,...
                'L', L...
                );
        end
        
        function output = forwardprop(obj, input)
            obj.input = input;
            
            [~, batchsize] = size(input.mu);
            output = zeros(obj.prms.J, batchsize, obj.prms.L);
            for l=1:(obj.prms.L)
                output(:, :, l) = input.mu + bsxfun(@times, sqrt(input.sig), obj.prms.eps(:, l));
            end
        end
        
        function delta = backwardprop(obj, input)
        end
        
        function init(obj)
            obj.prms.eps = mvnrnd(zeros(1, obj.prms.J), diag(ones(obj.prms.J, 1)), obj.prms.L)';
            
        end
        
        function update(obj)
        end
    end
end