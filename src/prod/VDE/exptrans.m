classdef exptrans < basenode
    properties
        input, prms, grad, optm
    end
    
    methods
        function obj = exptrans()
            obj.prms = struct();
            obj.optm = [];
        end
        
        function output = forwardprop(obj, input)
            obj.input = input;
            output = exp(input);
        end
        
        function delta = backwardprop(obj, input)
            delta = input.*exp(obj.input);
        end
        
        function init(obj)
        end
        
        function update(obj)
        end
        
        function refresh(obj)
        end
    end
end