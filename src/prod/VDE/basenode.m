classdef (Abstract) basenode < handle
    properties (Abstract)
        input, prms, delta
    end
    
    methods (Abstract)
        output = forwardprop(obj, input)
        delta = backwardprop(obj, input)
        init(obj)
        update(obj)
    end
end