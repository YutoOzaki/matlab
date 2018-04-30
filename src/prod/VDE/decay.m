classdef decay < optimizer
    properties
        eta, alpha, a, b, k
        dir
    end
    
    methods
        function obj = decay(alpha, a, b, dir)
            assert(alpha > 0.5 && alpha <= 1, 'alpha should be 0.5 < alpha <=1');
            
            obj.alpha = alpha;
            obj.a = a;
            obj.b = b;
            obj.k = struct();
            
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
        
        function init(obj, prms)
            prmnames = fieldnames(prms);
            
            for l=1:length(prmnames)
                if isa(prms.(prmnames{l}), 'gpuArray')
                    obj.k.(prmnames{l}) = rand(1, 1, 'single', 'gpuArray') * 0;
                else
                    obj.k.(prmnames{l}) = 0;
                end
            end
        end
        
        function updateval = adjust(obj, grad, prmname)
            obj.eta = obj.a/(obj.b + obj.k.(prmname))^(obj.alpha);
            updateval = obj.dir .* obj.eta.*grad;
            
            obj.k.(prmname) = obj.k.(prmname) + 1;
        end
        
        function refresh(obj)
            prmnames = fieldnames(obj.k);
            
            for l=1:length(prmnames)
                if isa(obj.k.(prmnames{l}), 'gpuArray')
                    obj.k.(prmnames{l}) = rand(1, 1, 'single', 'gpuArray') * 0;
                else
                    obj.k.(prmnames{l}) = 0;
                end
            end
        end
    end
end