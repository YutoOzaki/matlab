classdef BSoftmaxLayer < BaseLayer
    properties
        vis, hid, T, batchSize
        prms, states, gprms
        input, delta
    end
    
    properties (Constant)
        prmNum = 3;
        stateNum = 2;
        normInd = 1;
    end
    
    methods
        function myInit(obj)
            obj.delta = {obj.delta obj.delta};
        end
        
        function affineTrans(obj, x)
            obj.input = x;
            bMat = repmat(obj.prms{2}, 1, obj.batchSize);
            
            for t=1:obj.T
                obj.states{1}(:,:,t) = obj.prms{1}*x{1}(:,:,t) + obj.prms{3}*x{2}(:,:,t) + bMat;
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
        
        function continueStates(obj)
        end
        
        function resetStates(obj)
        end
        
        function dgate = bpropGate(obj, d)
            dgate = {d};
        end
        
        function delta = bpropDelta(obj, dgate)
            d = dgate{1};
            
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
            obj.states{1} = zeros(obj.hid, obj.batchSize, obj.T);   % v
            obj.states{2} = zeros(obj.hid, obj.batchSize, obj.T);   % y
        end
    end
end