classdef GradientChecker < handle
    properties
        L, batchNumCnt, batchNum, reLog
    end
    
    properties (Constant = true)
        meps = eps^(1/3)
    end
    
    methods
        function obj = GradientChecker(gcheck, L, batchNum, nnet)
            if gcheck
                obj.L = L;
            else
                obj.L = 0;
            end
            
            obj.batchNum = batchNum;
            obj.batchNumCnt = 1;
            
            obj.reLog = cell(1,L);
            for l=1:L
                if nnet{l}.BN
                    prmNum = nnet{l}.prmNum + 2*length(nnet{l}.normInd);
                    obj.reLog{1,l} = zeros(3, prmNum, batchNum);
                else
                    obj.reLog{1,l} = zeros(3, nnet{l}.prmNum, batchNum);
                end
            end
        end
        
        function gradientChecking(obj, nnet, dataMat, labelVector)
            for l=1:obj.L
                reBuf = obj.reLog{1,l};
                
                for k=1:nnet{l}.prmNum
                   val = nnet{l}.prms{k}(1,1);
                   h = max(abs(val),1) * obj.meps;

                   nnet{l}.prms{k}(1,1) = val + h;
                   input = dataMat;
                   for p=1:obj.L
                       input = nnet{p}.fprop(input);
                   end
                   dy1 = sum(sum(labelVector.*log(input),3),1);

                   nnet{l}.prms{k}(1,1) = val - h;
                   input = dataMat;
                   for p=1:obj.L
                       input = nnet{p}.fprop(input);
                   end
                   dy2 = sum(sum(labelVector.*log(input),3),1);

                   nnet{l}.prms{k}(1,1) = val;

                   dt1 = mean((-dy1 + dy2)./(2*h));
                   dt2 = nnet{l}.gprms{k}(1,1);
                   relativeError = abs(dt1 - dt2)/max(abs(dt1),abs(dt2));
                   reBuf(:,k,obj.batchNumCnt) = [relativeError,dt1,dt2];
                end
                
                if nnet{l}.BN
                    k_ = k;
                    
                    for k=1:length(nnet{l}.normInd)
                        for j=1:2
                            val = nnet{l}.BNprms{j,k}(1,1);
                            h = max(abs(val),1) * obj.meps;

                            nnet{l}.BNprms{j,k}(1,1) = val + h;
                            input = dataMat;
                            for p=1:obj.L
                                input = nnet{p}.fprop(input);
                            end
                            dy1 = sum(sum(labelVector.*log(input),3),1);

                            nnet{l}.BNprms{j,k}(1,1) = val - h;
                            input = dataMat;
                            for p=1:obj.L
                                input = nnet{p}.fprop(input);
                            end
                            dy2 = sum(sum(labelVector.*log(input),3),1);

                            nnet{l}.BNprms{j,k}(1,1) = val;

                            dt1 = mean((-dy1 + dy2)./(2*h));
                            dt2 = nnet{l}.BNgprms{j,k}(1,1);
                            relativeError = abs(dt1 - dt2)/max(abs(dt1),abs(dt2));
                            
                            ind = k_ + j + 2*(k-1);
                            
                            reBuf(:,ind,obj.batchNumCnt) = [relativeError,dt1,dt2];
                        end
                    end
                end
                
                obj.reLog{1,l} = reBuf;
            end
            
            obj.batchNumCnt = obj.batchNumCnt + 1;
            
            if obj.batchNumCnt > obj.batchNum
                obj.batchNumCnt = 1;
            end
        end
        
        function disp(obj)
            for l=1:obj.L
                subplot(obj.L,1,l);plot(log10(squeeze(obj.reLog{l}(1,:,:)))');ylim([-10 0]);
            end
            drawnow
        end
    end
end