classdef LSTMP < BaseLayer
    properties
        vis, hid, T, batchSize
        prms, states, gprms
        input, delta
    end
    
    properties (Constant)
        prmNum = 16;
        stateNum = 11;
        normInd = [1;5;6;7];
    end
    
    methods
        function affineTrans(obj, x)
            obj.input = x;
            
            b_zMat = repmat(obj.prms{9}, 1, obj.batchSize);
            b_fMat = repmat(obj.prms{10}, 1, obj.batchSize);
            b_iMat = repmat(obj.prms{11}, 1, obj.batchSize);
            b_oMat = repmat(obj.prms{12}, 1, obj.batchSize);
            
            for t=1:obj.T
                obj.states{1}(:,:,t) = obj.prms{1}*x(:,:,t) + b_zMat;
                obj.states{5}(:,:,t) = obj.prms{2}*x(:,:,t) + b_fMat;
                obj.states{6}(:,:,t) = obj.prms{3}*x(:,:,t) + b_iMat;
                obj.states{7}(:,:,t) = obj.prms{4}*x(:,:,t) + b_oMat;
            end
        end
        
        function output = nonlinearTrans(obj)
            % states: u, z, c, h, F, I, O, gF, gI, gO
            for t=1:obj.T
                obj.states{1}(:,:,t) = obj.states{1}(:,:,t) + obj.prms{5}*obj.states{11}(:,:,t);
                obj.states{2}(:,:,t) = tanh(obj.states{1}(:,:,t));

                obj.states{5}(:,:,t) = obj.states{5}(:,:,t) + obj.prms{6}*obj.states{11}(:,:,t) + obj.prms{13}*obj.states{3}(:,:,t);
                obj.states{8}(:,:,t) = obj.sigmoid(obj.states{5}(:,:,t));
                
                obj.states{6}(:,:,t) = obj.states{6}(:,:,t) + obj.prms{7}*obj.states{11}(:,:,t) + obj.prms{14}*obj.states{3}(:,:,t);
                obj.states{9}(:,:,t) = obj.sigmoid(obj.states{6}(:,:,t));

                obj.states{3}(:,:,t+1) = obj.states{2}(:,:,t).*obj.states{9}(:,:,t) + obj.states{3}(:,:,t).*obj.states{8}(:,:,t);
                
                obj.states{7}(:,:,t) = obj.states{7}(:,:,t) + obj.prms{8}*obj.states{11}(:,:,t) + obj.prms{15}*obj.states{3}(:,:,t+1);
                obj.states{10}(:,:,t) = obj.sigmoid(obj.states{7}(:,:,t));
                
                obj.states{4}(:,:,t+1) = tanh(obj.states{3}(:,:,t+1)) .* obj.states{10}(:,:,t);
                
                obj.states{11}(:,:,t+1) = obj.prms{16}*obj.states{4}(:,:,t+1);
            end
            
            output = obj.states{11}(:,:,2:end);
        end
        
        function dgate = bpropGate(obj, d)
            dz = repmat(obj.states{1}(:,:,1).*0, 1, 1, obj.T+1);
            dF = repmat(obj.states{5}(:,:,1).*0, 1, 1, obj.T+1);
            dI = repmat(obj.states{6}(:,:,1).*0, 1, 1, obj.T+1);
            dO = repmat(obj.states{7}(:,:,1).*0, 1, 1, obj.T+1);
            
            dc = obj.states{3}(:,:,1).*0;
            
            gradR_z = 0; gradR_o = 0; gradR_f = 0; gradR_i = 0;
            gradP_o = 0; gradP_f = 0; gradP_i = 0;
            gradR_p = 0;
            
            for t=obj.T:-1:1
                dh = d(:,:,t) + obj.prms{5}'*dz(:,:,t+1) + obj.prms{6}'*dF(:,:,t+1) + obj.prms{7}'*dI(:,:,t+1) + obj.prms{8}'*dO(:,:,t+1);
                
                gradR_p = gradR_p + dh*obj.states{4}(:,:,t+1)';
                
                dh = obj.prms{16}'*dh;
                
                dO(:,:,t) = dh .* tanh(obj.states{3}(:,:,t+1)) .* obj.dsigmoid(obj.states{7}(:,:,t));
                
                dc = dh.*obj.states{10}(:,:,t).*obj.dtanh(obj.states{3}(:,:,t+1))...
                    + obj.prms{13}*dF(:,:,t+1) + obj.prms{14}*dI(:,:,t+1) + obj.prms{15}*dO(:,:,t)...
                    + dc.*obj.states{8}(:,:,t+1);
                
                dF(:,:,t) = dc.*obj.states{3}(:,:,t) .* obj.dsigmoid(obj.states{5}(:,:,t));
                dI(:,:,t) = dc.*obj.states{2}(:,:,t) .* obj.dsigmoid(obj.states{6}(:,:,t));
                dz(:,:,t) = dc.*obj.states{9}(:,:,t) .* obj.dtanh(obj.states{1}(:,:,t));
                
                gradR_z = gradR_z + dz(:,:,t)*obj.states{11}(:,:,t)';
                gradR_f = gradR_f + dF(:,:,t)*obj.states{11}(:,:,t)';
                gradR_i = gradR_i + dI(:,:,t)*obj.states{11}(:,:,t)';
                gradR_o = gradR_o + dO(:,:,t)*obj.states{11}(:,:,t)';
                
                gradP_f = gradP_f + dF(:,:,t).*obj.states{3}(:,:,t);
                gradP_i = gradP_i + dI(:,:,t).*obj.states{3}(:,:,t);
                gradP_o = gradP_o + dO(:,:,t).*obj.states{3}(:,:,t+1);
            end
            
            obj.gprms{5} = gradR_z./obj.batchSize;
            obj.gprms{6} = gradR_f./obj.batchSize;
            obj.gprms{7} = gradR_i./obj.batchSize;
            obj.gprms{8} = gradR_o./obj.batchSize;
            
            obj.gprms{13} = diag(mean(gradP_f, 2));
            obj.gprms{14} = diag(mean(gradP_i, 2));
            obj.gprms{15} = diag(mean(gradP_o, 2));
            
            obj.gprms{16} = gradR_p./obj.batchSize;
            
            dgate = {dz, dF, dI, dO};
        end
        
        function delta = bpropDelta(obj, dgate)    
            dz = dgate{1};
            dF = dgate{2};
            dI = dgate{3};
            dO = dgate{4};
            
            gradW_z = 0; gradW_o = 0; gradW_f = 0; gradW_i = 0;
            gradb_z = 0; gradb_o = 0; gradb_f = 0; gradb_i = 0;
            
            for t=obj.T:-1:1
                obj.delta(:,:,t) = obj.prms{1}'*dz(:,:,t) + obj.prms{2}'*dF(:,:,t) + obj.prms{3}'*dI(:,:,t) + obj.prms{4}'*dO(:,:,t);

                gradW_z = gradW_z + dz(:,:,t)*obj.input(:,:,t)';
                gradW_f = gradW_f + dF(:,:,t)*obj.input(:,:,t)';
                gradW_i = gradW_i + dI(:,:,t)*obj.input(:,:,t)';
                gradW_o = gradW_o + dO(:,:,t)*obj.input(:,:,t)';

                gradb_z = gradb_z + dz(:,:,t);
                gradb_f = gradb_f + dF(:,:,t);
                gradb_i = gradb_i + dI(:,:,t);
                gradb_o = gradb_o + dO(:,:,t);
            end
            
            obj.gprms{1} = gradW_z./obj.batchSize;
            obj.gprms{2} = gradW_f./obj.batchSize;
            obj.gprms{3} = gradW_i./obj.batchSize;
            obj.gprms{4} = gradW_o./obj.batchSize;

            obj.gprms{9} = mean(gradb_z, 2);
            obj.gprms{10} = mean(gradb_f, 2);
            obj.gprms{11} = mean(gradb_i, 2);
            obj.gprms{12} = mean(gradb_o, 2);
            
            delta = obj.delta;
        end
        
        function initPrms(obj)
            obj.prms{1} = 2.*(rand(obj.hid, obj.vis) - 0.5) .* sqrt(6/(obj.vis+obj.hid));
            obj.prms{2} = 2.*(rand(obj.hid, obj.vis) - 0.5) .* sqrt(6/(obj.vis+obj.hid));
            obj.prms{3} = 2.*(rand(obj.hid, obj.vis) - 0.5) .* sqrt(6/(obj.vis+obj.hid));
            obj.prms{4} = 2.*(rand(obj.hid, obj.vis) - 0.5) .* sqrt(6/(obj.vis+obj.hid));
            
            obj.prms{5} = 2.*(rand(obj.hid, obj.hid) - 0.5) .* sqrt(6/(obj.hid+obj.hid));
            obj.prms{6} = 2.*(rand(obj.hid, obj.hid) - 0.5) .* sqrt(6/(obj.hid+obj.hid));
            obj.prms{7} = 2.*(rand(obj.hid, obj.hid) - 0.5) .* sqrt(6/(obj.hid+obj.hid));
            obj.prms{8} = 2.*(rand(obj.hid, obj.hid) - 0.5) .* sqrt(6/(obj.hid+obj.hid));
            
            obj.prms{9} = zeros(obj.hid, 1);
            obj.prms{10} = zeros(obj.hid, 1);
            obj.prms{11} = zeros(obj.hid, 1);
            obj.prms{12} = zeros(obj.hid, 1);
            
            obj.prms{13} = diag(2.*(rand(obj.hid, 1) - 0.5) .* sqrt(6/(obj.hid+1)));
            obj.prms{14} = diag(2.*(rand(obj.hid, 1) - 0.5) .* sqrt(6/(obj.hid+1)));
            obj.prms{15} = diag(2.*(rand(obj.hid, 1) - 0.5) .* sqrt(6/(obj.hid+1)));
            
            obj.prms{16} = 2.*(rand(obj.hid, obj.hid) - 0.5) .* sqrt(6/(obj.hid+obj.hid));
        end
        
        function initStates(obj)
            obj.states{1} = zeros(obj.hid, obj.batchSize, obj.T);   % u
            obj.states{2} = zeros(obj.hid, obj.batchSize, obj.T);   % z
            obj.states{3} = zeros(obj.hid, obj.batchSize, obj.T+1); % c
            obj.states{4} = zeros(obj.hid, obj.batchSize, obj.T+1); % h
            obj.states{5} = zeros(obj.hid, obj.batchSize, obj.T);   % F
            obj.states{6} = zeros(obj.hid, obj.batchSize, obj.T);   % I
            obj.states{7} = zeros(obj.hid, obj.batchSize, obj.T);   % O
            obj.states{8} = zeros(obj.hid, obj.batchSize, obj.T+1); % gF
            obj.states{9} = zeros(obj.hid, obj.batchSize, obj.T);   % gI
            obj.states{10} = zeros(obj.hid, obj.batchSize, obj.T);  % gO
            
            obj.states{11} = zeros(obj.hid, obj.batchSize, obj.T+1); % p (projection layer)
        end
    end
end