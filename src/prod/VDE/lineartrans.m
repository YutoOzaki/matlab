classdef lineartrans < basenode
    properties
        input, prms, grad, optm
    end
    
    methods
        function obj = lineartrans(J, D, optm)
            obj.prms = struct(...
                'W', zeros(J, D),...
                'b', zeros(J, 1)...
            );
            
            obj.optm = optm;
        end
        
        function output = forwardprop(obj, input)
            obj.input = input;
            buf = obj.prms.W * input;
            output = bsxfun(@plus, buf, obj.prms.b);
        end
        
        function delta = backwardprop(obj, input)
            batchsize = size(obj.input, 2);
            
            gb = sum(input, 2)./batchsize;
            gW = obj.input * input'./batchsize;
            
            obj.grad = struct(...
                'b', gb,...
                'W', gW'...
                );
            
            delta = obj.prms.W' * input;
        end
        
        function init(obj)
            [J, D] = size(obj.prms.W);
            r = sqrt(6/(J + D));
            obj.prms.W = r.*2.*(rand(J, D) - 0.5);
        end
        
        function update(obj)
            prmnames = fieldnames(obj.grad);
            
            for l=1:length(prmnames)
                obj.prms.(prmnames{l}) = obj.prms.(prmnames{l}) + obj.optm.adjust(obj.grad.(prmnames{l}));
            end
        end
    end
end