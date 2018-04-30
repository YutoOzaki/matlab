classdef reparamtrans < basenode
    properties
        input, prms, grad, optm
        eps, J, L
        poolsize, counter % pool random variables to avoid call mvnrnd() every time
        gpuwrapper
    end
    
    methods
        function obj = reparamtrans(J, L, poolsize, gpumode)
            obj.prms = struct();
            obj.J = J;
            obj.L = L;
            obj.poolsize = poolsize;
            obj.optm = [];
            
            switch nargin
                case 4
                    if gpumode
                        obj.gpuwrapper = @(x) gpuArray(cast(x, 'single'));
                    else
                        obj.gpuwrapper = @(x) x;
                    end
                otherwise
                    obj.gpuwrapper = @(x) x;
            end
        end
        
        function output = forwardprop(obj, input)
            obj.input = input;
            
            [~, batchsize] = size(input.mu);
            output = obj.gpuwrapper(zeros(obj.J, batchsize, obj.L));
            
            idx = obj.counter*obj.L;
            for l=1:(obj.L)
                output(:, :, l) = input.mu + bsxfun(@times, sqrt(input.sig), obj.eps(:, idx+l));
            end
        end
        
        function delta = backwardprop(obj, input)
            dzdmu = sum(input, 3);
            
            dzdsig = 0;
            idx = obj.counter*obj.L;
            for l=1:obj.L
                A = bsxfun(@times, input(:, :, l), obj.eps(:, idx+l));
                
                dzdsig = dzdsig + 0.5.*A./sqrt(obj.input.sig);
            end
            
            delta = struct(...
                'mu', dzdmu,...
                'sig', dzdsig...
                );
        end
        
        function init(obj)
            obj.eps = obj.gpuwrapper(mvnrnd(zeros(1, obj.J), diag(ones(obj.J, 1)), obj.L*obj.poolsize))';
            obj.counter = 0;
        end
        
        function update(obj)
            obj.counter = obj.counter + 1;
            
            if obj.counter == obj.poolsize
                obj.counter = 0;
            end
        end
        
        function refresh(obj)
        end
    end
end