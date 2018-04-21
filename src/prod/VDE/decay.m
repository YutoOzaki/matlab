classdef decay < optimizer
    properties
        eta, alpha, a, b, k
        dir, ms
    end
    
    methods
        function obj = decay(alpha, a, b, dir)
            assert(alpha > 0.5 && alpha <= 1, 'alpha should be 0.5 < alpha <=1');
            
            obj.alpha = alpha;
            obj.a = a;
            obj.b = b;
            obj.k = 0;
            
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
            %obj.eta = (2 + obj.k)^(-obj.a);
            obj.eta = obj.a/(obj.b + obj.k)^(obj.alpha);
            updateval = obj.dir .* obj.eta.*grad;
            
            obj.k = obj.k + 1;
        end
        
        function refresh(obj)
        end
    end
end