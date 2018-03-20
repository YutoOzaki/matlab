classdef exptrans < basenode
    properties
        input, prms, grad
    end
    
    methods
        function obj = exptrans()
            obj.prms = [];
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
    end
end