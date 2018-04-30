classdef optimizer < handle
    methods (Abstract)
        updateval = adjust(obj, grad);
        refresh(obj);
        init(obj, prms);
    end
end