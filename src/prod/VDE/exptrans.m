classdef exptrans < basenode
    properties
        input, prms, delta
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
        end
        
        function init(obj)
        end
        
        function update(obj)
        end
    end
end