classdef relutrans < basenode
    properties
        input, prms, grad, optm
    end
    
    methods
        function obj = relutrans()
            obj.prms = struct();
            obj.optm = [];
        end
        
        function output = forwardprop(obj, input)
            obj.input = input;
            idx = input < 0;
            input(idx) = 0;
            output = input;
        end
        
        function delta = backwardprop(obj, input)
            idx = obj.input < 0;
            dinput = ones(size(input, 1), size(input, 2), class(input));
            dinput(idx) = 0;
            
            delta = input.*dinput;
        end
        
        function init(obj)
        end
        
        function update(obj)
        end
        
        function refresh(obj)
        end
    end
end