classdef adam < optimizer
    properties
        b1, b2, eta, e, ms, mg, t
        dir
    end
    
    methods
        function obj = adam(b1, b2, eta, e, dir)
            obj.b1 = b1;
            obj.b2 = b2;
            obj.eta = eta;
            obj.e = e;
            obj.ms = struct();
            obj.mg = struct();
            obj.t = struct();
            
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
            newmg = obj.b1.*obj.mg.(prmname) + (1 - obj.b1).*grad;
            newms = obj.b2.*obj.ms.(prmname) + (1 - obj.b2).*(grad.^2);
            
            newt = obj.t.(prmname) + 1;
            
            mghat = newmg./(1 - obj.b1^newt);
            mshat = newms./(1 - obj.b2^newt);
            
            updateval = obj.dir .* obj.eta.*mghat./(sqrt(mshat) + obj.e);
            
            obj.mg.(prmname) = newmg;
            obj.ms.(prmname) = newms;
            obj.t.(prmname) = newt;
        end
        
        function init(obj, prms)
            prmnames = fieldnames(prms);
            
            for i=1:length(prmnames)
                obj.ms.(prmnames{i}) = prms.(prmnames{i}).*0;
                obj.mg.(prmnames{i}) = prms.(prmnames{i}).*0;
                obj.t.(prmnames{i}) = 0;
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