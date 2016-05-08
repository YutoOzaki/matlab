classdef BaseLayer < handle
    properties (Abstract = true)
        vis, hid, T, batchSize
        prms, states, gprms
        input, delta
    end
    
    properties
        updatePrms, updateFun
        
        BNprms, BNstates, BNgprms, BNupdatePrms
        BN, BNprmNum
        bneps = 1e-6;
        
        dropRate, dropMask, compensationFlag = true;
        
        layer
        
        sigmoid = @(x) 1./(1 + exp(-x));
        dsigmoid = @(x) (1./(1 + exp(-x))).*(1 - 1./(1 + exp(-x)));
        dtanh = @(x) 1 - tanh(x).^2;
    end
    
    properties (Abstract = true, Constant = true)
        prmNum, stateNum, 
        normInd % index of state to be batch-normalized
    end
    
    methods (Abstract = true)
        affineTrans(obj, x)
        nonlinearTrans(obj)
        bpropGate(obj, d)
        bpropDelta(obj, dgate)
        continueStates(obj)
        resetStates(obj)
        initPrms(obj)
        initStates(obj)
    end
    
    methods        
        function initLayer(obj, vis, hid, T, batchSize, isBN, dropRate)
            persistent L;
            if isempty(L)
                L = 1;
            end
            obj.layer = L;
            L = L + 1;
            
            
            obj.vis = vis;
            obj.hid = hid;
            obj.T = T;
            obj.batchSize = batchSize;
            
            obj.prms = cell(obj.prmNum, 1);
            
            obj.states = cell(obj.stateNum,1);
            
            obj.delta = zeros(vis, batchSize, T);
            
            initPrms(obj);
            initStates(obj);
            
            obj.gprms = obj.prms;
            
            if nargin < 7
                obj.dropRate = 0.0;
                createDropMask(obj);
                
                if nargin < 6
                    obj.BN = false;
                else
                    initBNLayer(obj, isBN);
                end
            else
                obj.dropRate = dropRate;
                obj.BN = isBN;
                initBNLayer(obj, isBN);
            end
            
            myInit(obj);
        end
        
        function myInit(obj)
        end
        
        function initBNLayer(obj, BN)
            if BN
                obj.BNstates = cell(4,length(obj.normInd));
                
                obj.BNprms = cell(2,length(obj.normInd));
                obj.BNprms(1,:) = {ones(obj.hid,1)};
                obj.BNprms(2,:) = {zeros(obj.hid,1)};
                
                obj.BNgprms = cell(2, length(obj.normInd));
                
                obj.BNprmNum = length(obj.normInd) * 2; % scale and shift for each gradient channel
            end
            
            obj.BN = BN;
        end
        
        function gradNorm(obj)
            gnorm = cell2mat(cellfun(@(x) norm(x,2), obj.gprms, 'UniformOutput', false));
            denom = cell2mat(cellfun(@(x) numel(x), obj.gprms, 'UniformOutput', false));
            
            figure(obj.layer)
            subplot(211); plot(gnorm);
            subplot(212); plot(gnorm./denom);
            drawnow
        end
        
        function output = fprop(obj, x)
            if iscell(x)
                x{1} = x{1}.*obj.dropMask;
                x{2} = x{2}.*obj.dropMask;
            else
                x = x.*obj.dropMask;
            end
            
            affineTrans(obj, x);
            if obj.BN
                batchNormalization(obj);
            end
            output = nonlinearTrans(obj);
        end
        
        function delta = bprop(obj, d)
            dgate = bpropGate(obj, d);
            if obj.BN
                dgate = bpropBN(obj, dgate);
            end
            delta = bpropDelta(obj, dgate);
            
            if iscell(delta)
                delta{1} = delta{1}.*obj.dropMask;
                delta{2} = delta{2}.*obj.dropMask;
            else
                delta = delta.*obj.dropMask;
            end
        end
        
        function createDropMask(obj)
            obj.dropMask = binornd(1, 1 - obj.dropRate, obj.vis, obj.batchSize, obj.T);
        end
        
        function dropoutCompensation(obj)
            if obj.compensationFlag
                factor = 1 - obj.dropRate;
                obj.dropMask = binornd(1, 1, obj.vis, obj.batchSize, obj.T);
            else
                factor = 1./(1 - obj.dropRate);
            end
            
            obj.compensationFlag = not(obj.compensationFlag);
            
            obj.prms = cellfun(@(x) x.*factor, obj.prms, 'UniformOutput', false);
        end
        
        function batchNormalization(obj)
            for i=1:length(obj.normInd)
                k = obj.normInd(i);
                obj.BNstates{1,i} = obj.states{k};
                
                obj.BNstates{2,i} = mean(mean(obj.BNstates{1,i},3),2);
                muMat = repmat(obj.BNstates{2,i}, 1, obj.batchSize, obj.T);
                
                obj.BNstates{3,i} = sum(sum((obj.BNstates{1,i} - muMat).^2,3),2)./(obj.T*obj.batchSize);
                sigMat = repmat(obj.BNstates{3,i}, 1, obj.batchSize, obj.T);
                
                obj.BNstates{4,i} = (obj.BNstates{1,i} - muMat)./sqrt(sigMat + obj.bneps);
                
                obj.states{k} = repmat(obj.BNprms{1,i}, 1, obj.batchSize, obj.T)...
                    .* obj.BNstates{4,i}...
                    + repmat(obj.BNprms{2,i}, 1, obj.batchSize, obj.T);
            end
        end
        
        function dgate = bpropBN(obj, dgate)
            N = obj.batchSize*obj.T;

            for i=1:length(obj.normInd)
                obj.BNgprms{1,i} = sum(sum(dgate{i}(:,:,1:obj.T).*obj.BNstates{4,i}, 3), 2)./obj.batchSize;
                obj.BNgprms{2,i} = sum(sum(dgate{i}(:,:,1:obj.T), 3), 2)./obj.batchSize;
                
                dxhat = dgate{i}(:,:,1:obj.T).*repmat(obj.BNprms{1,i},1,obj.batchSize,obj.T);

                xmurep = obj.BNstates{1,i} - repmat(obj.BNstates{2,i},1,obj.batchSize,obj.T);

                dvar = sum(sum(...
                    repmat(-0.5.*(obj.BNstates{3,i} + obj.bneps).^(-3/2),1,obj.batchSize,obj.T)...
                    .*xmurep.*dxhat,3),2);

                ivarrep = repmat(1./sqrt(obj.BNstates{3,i} + obj.bneps),1,obj.batchSize,obj.T);

                dmu = sum(sum(-ivarrep.* dxhat,3),2);

                dgate{i} = dxhat.*ivarrep...
                    + repmat(dvar,1,obj.batchSize,obj.T).*2.*xmurep./N...
                    + repmat(dmu,1,obj.batchSize,obj.T)./N;
            end
        end
        
        function optimization(obj, method, lprm)
            uprmsBuf = cellfun(@(x) x.*0, obj.prms, 'UniformOutput', false);
            
            if obj.BN
                BNuprmsBuf = cellfun(@(x) x.*0, reshape(obj.BNprms,1,obj.BNprmNum), 'UniformOutput', false);
            end
            
            if strcmp(method,'rmsProp')
                obj.updatePrms = uprmsBuf;
                obj.updateFun = @(prms,gprms,rms) rmsProp(prms,gprms,rms,lprm(1),lprm(2),lprm(3));
                
                if obj.BN
                    obj.BNupdatePrms = BNuprmsBuf;
                end
            elseif strcmp(method,'adaDelta')
                obj.updatePrms = {uprmsBuf uprmsBuf};
                obj.updateFun = @(prms,gprms,exps) adaDelta(prms,gprms,exps,lprm(1),lprm(2));
                
                if obj.BN
                    obj.BNupdatePrms = {BNuprmsBuf BNuprmsBuf};
                end
            elseif strcmp(method,'adaGrad')
                obj.updatePrms = cellfun(@(x) x + lprm(2), uprmsBuf, 'UniformOutput', false);
                obj.updateFun = @(prms,gprms,grads) adaGrad(prms,gprms,grads,lprm(1));
                
                if obj.BN
                    obj.BNupdatePrms = cellfun(@(x) x + lprm(2), BNuprmsBuf, 'UniformOutput', false);
                end
            elseif strcmp(method,'adam')
                obj.updatePrms = {uprmsBuf uprmsBuf 1};
                obj.updateFun = @(prms,gprms,moments) adam(prms,gprms,moments,lprm(1),lprm(2),lprm(3),lprm(4));
                
                if obj.BN
                    obj.BNupdatePrms = {BNuprmsBuf BNuprmsBuf 1};
                end
            end
        end
        
        function update(obj)
            gradNorm(obj);
            [prms_new, updatePrms_new] = obj.updateFun(obj.prms, obj.gprms, obj.updatePrms);
            
            obj.prms = prms_new;
            obj.updatePrms = updatePrms_new;
            
            if obj.BN
                [prms_new, updatePrms_new] = ...
                    obj.updateFun(reshape(obj.BNprms,1,obj.BNprmNum), reshape(obj.BNgprms,1,obj.BNprmNum), obj.BNupdatePrms);
                
                obj.BNprms = reshape(prms_new,2,length(obj.normInd));
                obj.BNupdatePrms = updatePrms_new;
            end
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