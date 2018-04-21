classdef (Abstract) basenode < handle
    properties (Abstract)
        input, prms, grad, optm
    end
    
    methods (Abstract)
        output = forwardprop(obj, input)
        delta = backwardprop(obj, input)
        init(obj)
        update(obj)
        refresh(obj)
    end
end