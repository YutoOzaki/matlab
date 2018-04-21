classdef lineartrans < basenode
    properties
        input, prms, grad, optm
        weidec
    end
    
    methods
        function obj = lineartrans(J, D, optm, weidec)
            obj.prms = struct(...
                'W', zeros(J, D),...
                'b', zeros(J, 1)...
            );
            
            optm.ms = obj.prms;
            obj.optm = optm;
            
            switch nargin
                case 3
                    obj.weidec = 0;
                case 4
                    obj.weidec = weidec;
            end
        end
        
        function output = forwardprop(obj, input)
            obj.input = input;
            buf = obj.prms.W * input;
            output = bsxfun(@plus, buf, obj.prms.b);
        end
        
        function delta = backwardprop(obj, input)
            batchsize = size(obj.input, 2);
            
            gb = sum(input, 2)./batchsize;
            gW = obj.input * input'./batchsize + obj.weidec.*obj.prms.W';
            
            obj.grad = struct(...
                'b', gb,...
                'W', gW'...
                );
            
            delta = obj.prms.W' * input;
        end
        
        function init(obj)
            [J, D] = size(obj.prms.W);
            r = sqrt(6/(J + D));
            obj.prms.W = r.*2.*(rand(J, D) - 0.5);
        end
        
        function update(obj)
            prmnames = fieldnames(obj.grad);
            
            for l=1:length(prmnames)
                obj.prms.(prmnames{l}) = obj.prms.(prmnames{l}) + obj.optm.adjust(obj.grad.(prmnames{l}), prmnames{l});
            end
        end
        
        function refresh(obj)
            obj.weidec = 0;
            obj.optm.refresh();
        end
        
        function setoptm(obj, optm)
            obj.optm = optm;
            obj.optm.ms = obj.prms;
            obj.optm.refresh();
        end
    end
end