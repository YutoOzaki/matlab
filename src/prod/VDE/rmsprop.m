classdef rmsprop < optimizer
    properties
        r, a, e, ms
        dir
    end
    
    methods
        function obj = rmsprop(r, a, e, dir)
            obj.r = r;
            obj.a = a;
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
            obj.ms.(prmname) = obj.r.*obj.ms.(prmname) + (1 - obj.r).*(grad.^2);
            updateval = obj.dir .* obj.a.*grad./(sqrt(obj.ms.(prmname)) + obj.e);
        end
        
        function init(obj, prms)
            prmnames = fieldnames(prms);
            
            for i=1:length(prmnames)
                obj.ms.(prmnames{i}) = prms.(prmnames{i}).*0;
            end
        end
        
        function refresh(obj)
            names = fieldnames(obj.ms);
            
            for i=1:length(names)
                obj.ms.(names{i}) = obj.ms.(names{i}).*0;
            end
        end
    end
end