classdef lineartrans < basenode
    properties
        input, prms, grad, optm
        weidec
    end
    
    methods
        function obj = lineartrans(J, D, optm, weidec, gpumode)
            W = zeros(J, D);
            b = zeros(J, 1);
            
            switch nargin
                case 3
                    obj.weidec = 0;
                case 4
                    obj.weidec = weidec;
                case 5
                    if gpumode
                        W = gpuArray(cast(W, 'single'));
                        b = gpuArray(cast(b, 'single'));
                        weidec = gpuArray(weidec);
                    end
                    obj.weidec = weidec;
            end
            
            obj.prms = struct(...
                'W', W,...
                'b', b...
            );
        
            obj.grad = struct(...
                'W', W.*0,...
                'b', b.*0 ...
            );
            
            obj.optm = optm;
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
            
            obj.grad.b = gb;
            obj.grad.W = gW';
            
            delta = obj.prms.W' * input;
        end
        
        function init(obj)
            [J, D] = size(obj.prms.W);
            r = sqrt(6/(J + D));
            W = r.*2.*(rand(J, D) - 0.5);
            
            if isa(obj.prms.W, 'gpuArray')
                W = gpuArray(cast(W, 'single'));
            end
            
            obj.prms.W = W;
            
            obj.optm.init(obj.prms);
        end
        
        function update(obj)
            prmnames = fieldnames(obj.grad);
            
            for l=1:length(prmnames)
                newprm = obj.prms.(prmnames{l}) + obj.optm.adjust(obj.grad.(prmnames{l}), prmnames{l});
                obj.prms.(prmnames{l}) = newprm;
            end
        end
        
        function refresh(obj)
            obj.weidec = 0;
            obj.optm.refresh();
        end
        
        function setoptm(obj, optm)
            obj.optm = optm;
            obj.optm.init(obj.prms);
        end
    end
end