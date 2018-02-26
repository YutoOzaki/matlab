classdef lineartrans < basenode
    properties
        input, prms, delta
    end
    
    methods
        function obj = lineartrans(J, D)
            obj.prms = struct(...
                'W', zeros(J, D),...
                'b', zeros(J, 1)...
            );
        end
        
        function output = forwardprop(obj, input)
            obj.input = input;
            buf = obj.prms.W * input;
            output = bsxfun(@plus, buf, obj.prms.b);
        end
        
        function delta = backwardprop(obj, input)
        end
        
        function init(obj)
            [J, D] = size(obj.prms.W);
            r = sqrt(6/(J + D));
            obj.prms.W = r.*2.*(rand(J, D) - 0.5);
        end
        
        function update(obj)
        end
    end
end