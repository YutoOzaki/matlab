classdef optimizer < handle
    methods (Abstract)
        updateval = adjust(obj, grad);
        refresh(obj);
    end
end