classdef adagrad < optimizer
    properties
        eta, e, ms
        dir
    end
    
    methods
        function obj = adagrad(eta, e, dir)
            obj.eta = eta;
            obj.e = e;
            obj.ms = struct();
            
            switch dir
                case 'asc'
                    obj.dir = 1;
                case 'desc'
                    obj.dir = -1;
            end
        end
        
        function direction(obj, dir)
            switch dir
                case 'asc'
                    obj.dir = 1;
                case 'desc'
                    obj.dir = -1;
            end
        end
        
        function updateval = adjust(obj, grad, prmname)
            obj.ms.(prmname) = obj.ms.(prmname) + grad.^2;
            updateval = obj.dir .* obj.eta.*grad./sqrt(obj.ms.(prmname) + obj.e);
        end
        
        function refresh(obj)
            names = fieldnames(obj.ms);
            
            for i=1:length(names)
                obj.ms.(names{i}) = obj.ms.(names{i}).*0;
            end
        end
    end
end