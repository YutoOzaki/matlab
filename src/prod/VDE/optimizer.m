classdef optimizer < handle
    methods (Abstract)
        updateval = adjust(obj, grad);
    end
end