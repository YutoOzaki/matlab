classdef rmsprop < optimizer
    properties
        a, r, e, ms
    end
    
    methods
        function obj = rmsprop(a, r, e)
            obj.a = a;
            obj.r = r;
            obj.e = e;
            obj.ms = 0;
        end
        
        function updateval = adjust(obj, grad)
            obj.ms = obj.r.*obj.ms + (1 - obj.r).*(grad.^2);
            updateval = -obj.a.*grad./(obj.ms + obj.e);
        end
    end
end