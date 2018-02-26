classdef tanhtrans < basenode
    properties
        input, prms, delta
    end
    
    methods
        function obj = tanhtrans()
            obj.prms = [];
        end
        
        function output = forwardprop(obj, input)
            obj.input = input;
            output = tanh(input);
        end
        
        function delta = backwardprop(obj, input)
        end
        
        function init(obj)
        end
        
        function update(obj)
        end
    end
end