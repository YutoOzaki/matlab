classdef tanhtrans < basenode
    properties
        input, prms, grad, optm
    end
    
    methods
        function obj = tanhtrans()
            obj.prms = struct();
            obj.optm = [];
        end
        
        function output = forwardprop(obj, input)
            obj.input = input;
            output = tanh(input);
        end
        
        function delta = backwardprop(obj, input)
            delta = input.*(1 - tanh(obj.input).^2);
        end
        
        function init(obj)
        end
        
        function update(obj)
        end
        
        function refresh(obj)
        end
    end
end