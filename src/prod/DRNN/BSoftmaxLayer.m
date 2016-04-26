classdef BSoftmaxLayer < BaseLayer
    properties
        vis, hid, T, batchSize
        prms, states, gprms, updatePrms
        input, delta
        updateFun
    end
    
    properties (Constant)
        prmNum = 3;
        stateNum = 1;
    end
    
    methods
        function obj = BSoftmaxLayer(vis, hid, T, batchSize)
            initLayer(obj, vis, hid, T, batchSize);
            obj.delta = {obj.delta obj.delta};
        end
        
        function output = fprop(obj, x)
            obj.input = x;
            bMat = repmat(obj.prms{2}, 1, obj.batchSize);
            
            for t=1:obj.T
                v = obj.prms{1}*x{1}(:,:,t) + obj.prms{3}*x{2}(:,:,t) + bMat;
                K = max(v);
                v = v - repmat(K, obj.hid, 1);
                obj.states{1}(:,:,t) = exp(v)./repmat(sum(exp(v)),obj.hid,1);
            end
            
            output = obj.states{1};
        end
        
        function delta = bprop(obj, d)
            gradWf = 0;
            gradWb = 0;
            gradb = 0;
            
            for t=obj.T:-1:1
                gradWf = gradWf + d(:,:,t) * obj.input{1}(:,:,t)';
                gradWb = gradWb + d(:,:,t) * obj.input{2}(:,:,t)';
                gradb = gradb + d(:,:,t);

                obj.delta{1}(:,:,t) = obj.prms{1}'*d(:,:,t);
                obj.delta{2}(:,:,t) = obj.prms{3}'*d(:,:,t);
            end
            
            obj.gprms{1} = gradWf./obj.batchSize;
            obj.gprms{2} = mean(gradb, 2);
            obj.gprms{3} = gradWb./obj.batchSize;
            
            delta = obj.delta;
        end
        
        function initPrms(obj)
            obj.prms{1} = 2.*(rand(obj.hid, obj.vis) - 0.5) .* sqrt(6/(obj.vis+obj.hid));
            obj.prms{2} = 2.*(rand(obj.hid,1) - 0.5) .* sqrt(6/obj.hid);
            obj.prms{3} = 2.*(rand(obj.hid, obj.vis) - 0.5) .* sqrt(6/(obj.vis+obj.hid));
        end
        
        function initStates(obj)
            obj.states{1} = zeros(obj.hid, obj.batchSize, obj.T);   % y
        end
    end
end