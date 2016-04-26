classdef BaseLayer < handle
    properties (Abstract = true)
        vis, hid, T, batchSize
        prms, states, gprms, updatePrms, BNPrms
        input, delta
        updateFun
    end
    
    properties
        BN
    end
    
    properties (Abstract = true, Constant = true)
        prmNum, stateNum, normInd
    end
    
    properties
        sigmoid = @(x) 1./(1 + exp(-x));
        dsigmoid = @(x) sigmoid(x).*(1 - sigmoid(x));
        dtanh = @(x) 1 - tanh(x).^2;
    end
    
    methods (Abstract = true)
        affineTrans(obj, x)
        nonlinearTrans(obj)
        bprop(obj, d)
        initPrms(obj)
        initStates(obj)
    end
    
    methods        
        function initLayer(obj, vis, hid, T, batchSize)            
            obj.vis = vis;
            obj.hid = hid;
            obj.T = T;
            obj.batchSize = batchSize;
            
            obj.prms = cell(obj.prmNum, 1);
            obj.gprms = cell(obj.prmNum, 1);
            
            obj.states = cell(obj.stateNum,1);
            
            obj.delta = zeros(vis, batchSize, T);
            
            obj.BN = false;
            
            initPrms(obj);
            initStates(obj);
        end
        
        function isBN(obj, BN)
            if BN
                obj.BNPrms = cell(5,length(obj.normInd));
                obj.BNPrms(3,:) = {zeros(obj.hid,1)};
                obj.BNPrms(4,:) = {ones(obj.hid,1)};
            end
            
            obj.BN = BN;
        end
        
        function output = fprop(obj, x)
            affineTrans(obj, x);
            if obj.BN
                batchNormalization(obj);
            end
            output = nonlinearTrans(obj);
        end
        
        function batchNormalization(obj)
            for i=1:length(obj.normInd)
                k = obj.normInd(i);
                obj.BNPrms{5,i} = obj.states{k};
                
                obj.BNPrms{1,i} = mean(mean(obj.states{k},3),2);
                muMat = repmat(obj.BNPrms{1,i}, 1, obj.batchSize, obj.T);
                obj.BNPrms{2,i} = sum(sum((obj.states{k} - muMat).^2,3),2)./(obj.T*obj.batchSize);
                sigMat = repmat(obj.BNPrms{2,i}, 1, obj.batchSize, obj.T);
                
                obj.states{k} = repmat(obj.BNPrms{4,i},1,obj.batchSize,obj.T)...
                    .* (obj.states{k} - muMat)./sqrt(sigMat)...
                    + repmat(obj.BNPrms{3,i},1,obj.batchSize,obj.T);
            end
        end
        
        function optimization(obj, method, lprm)
            if strcmp(method,'rmsProp')
                obj.updatePrms = cellfun(@(x) x.*0, obj.prms, 'UniformOutput', false);
                obj.updateFun = @(prms,gprms,rms) rmsProp(prms,gprms,rms,lprm(1),lprm(2),1-lprm(2),lprm(3),obj.prmNum);
            end
        end
        
        function update(obj)
            [prms_new, updatePrms_new] = obj.updateFun(obj.prms, obj.gprms, obj.updatePrms);
            
            obj.prms(1:obj.prmNum) = prms_new(1:obj.prmNum);
            obj.updatePrms(1:obj.prmNum) = updatePrms_new(1:obj.prmNum);
        end
        
        function onGPU(obj, ongpu)
            if ongpu
                for i=1:obj.prmNum
                    obj.prms{i} = gpuArray(obj.prms{i});
                end

                for i=1:obj.stateNum
                    obj.states{i} = gpuArray(obj.states{i});
                end
                
                for i=1:length(obj.delta)
                    obj.delta{i} = gpuArray(obj.delta{i});
                end
            end
        end
    end
end