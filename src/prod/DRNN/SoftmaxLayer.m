classdef SoftmaxLayer < BaseLayer
    properties
        vis, hid, T, batchSize
        prms, states, gprms, updatePrms, BNPrms
        input, delta
        updateFun
    end
    
    properties (Constant)
        prmNum = 2;
        stateNum = 2;
        normInd = 1;
    end
    
    methods
        function affineTrans(obj, x)
            obj.input = x;
            bMat = repmat(obj.prms{2}, 1, obj.batchSize);
            
            for t=1:obj.T
                obj.states{1}(:,:,t) = obj.prms{1}*x(:,:,t) + bMat;
            end
        end
        
        function output = nonlinearTrans(obj)
            for t=1:obj.T
                K = max(obj.states{1}(:,:,t));
                v = obj.states{1}(:,:,t) - repmat(K, obj.hid, 1);
                obj.states{2}(:,:,t) = exp(v)./repmat(sum(exp(v)),obj.hid,1);
            end
            
            output = obj.states{2};
        end
        
        function delta = bprop(obj, d)
            gradW = 0;
            gradb = 0;
            
            for t=obj.T:-1:1
                gradW = gradW + d(:,:,t) * obj.input(:,:,t)';
                gradb = gradb + d(:,:,t);

                obj.delta(:,:,t) = obj.prms{1}'*d(:,:,t);
            end
            
            obj.gprms{1} = gradW./obj.batchSize;
            obj.gprms{2} = mean(gradb,2);
            
            delta = obj.delta;
        end
        
        function initPrms(obj)
            obj.prms{1} = 2.*(rand(obj.hid, obj.vis) - 0.5) .* sqrt(6/(obj.vis+obj.hid));
            obj.prms{2} = 2.*(rand(obj.hid,1) - 0.5) .* sqrt(6/obj.hid);
        end
        
        function initStates(obj)
            obj.states{1} = zeros(obj.hid, obj.batchSize, obj.T);   % v
            obj.states{2} = zeros(obj.hid, obj.batchSize, obj.T);   % y
        end
    end
end