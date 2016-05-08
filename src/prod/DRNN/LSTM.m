classdef LSTM < BaseLayer
    properties
        vis, hid, T, batchSize
        prms, states, gprms
        input, delta
    end
    
    properties (Constant)
        prmNum = 15;
        stateNum = 10;
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
                obj.states{1}(:,:,t) = obj.states{1}(:,:,t) + obj.prms{5}*obj.states{4}(:,:,t);
                obj.states{2}(:,:,t) = tanh(obj.states{1}(:,:,t));

                obj.states{5}(:,:,t) = obj.states{5}(:,:,t) + obj.prms{6}*obj.states{4}(:,:,t) + obj.prms{13}*obj.states{3}(:,:,t);
                obj.states{8}(:,:,t) = obj.sigmoid(obj.states{5}(:,:,t));
                
                obj.states{6}(:,:,t) = obj.states{6}(:,:,t) + obj.prms{7}*obj.states{4}(:,:,t) + obj.prms{14}*obj.states{3}(:,:,t);
                obj.states{9}(:,:,t) = obj.sigmoid(obj.states{6}(:,:,t));

                obj.states{3}(:,:,t+1) = obj.states{2}(:,:,t).*obj.states{9}(:,:,t) + obj.states{3}(:,:,t).*obj.states{8}(:,:,t);
                
                obj.states{7}(:,:,t) = obj.states{7}(:,:,t) + obj.prms{8}*obj.states{4}(:,:,t) + obj.prms{15}*obj.states{3}(:,:,t+1);
                obj.states{10}(:,:,t) = obj.sigmoid(obj.states{7}(:,:,t));
                
                obj.states{4}(:,:,t+1) = tanh(obj.states{3}(:,:,t+1)) .* obj.states{10}(:,:,t);
            end
            
            output = obj.states{4}(:,:,2:end);
        end
        
        function continueStates(obj)
            obj.states{4}(:,:,1) = obj.states{4}(:,:,end);
        end
        
        function resetStates(obj)
            obj.states{4} = obj.states{4}.*0;
        end
        
        function dgate = bpropGate(obj, d)
            dz = repmat(obj.states{1}(:,:,1).*0, 1, 1, obj.T+1);
            dF = repmat(obj.states{5}(:,:,1).*0, 1, 1, obj.T+1);
            dI = repmat(obj.states{6}(:,:,1).*0, 1, 1, obj.T+1);
            dO = repmat(obj.states{7}(:,:,1).*0, 1, 1, obj.T+1);
            
            dc = obj.states{3}(:,:,1).*0;
            
            gradR_z_tmp = 0; gradR_o_tmp = 0; gradR_f_tmp = 0; gradR_i_tmp = 0;
            gradP_o_tmp = 0; gradP_f_tmp = 0; gradP_i_tmp = 0;
            
            for t=obj.T:-1:1
                dh = d(:,:,t) + obj.prms{5}'*dz(:,:,t+1) + obj.prms{6}'*dF(:,:,t+1) + obj.prms{7}'*dI(:,:,t+1) + obj.prms{8}'*dO(:,:,t+1);
                
                dO(:,:,t) = dh .* tanh(obj.states{3}(:,:,t+1)) .* obj.dsigmoid(obj.states{7}(:,:,t));
                
                dc = dh.*obj.states{10}(:,:,t).*obj.dtanh(obj.states{3}(:,:,t+1))...
                    + obj.prms{13}*dF(:,:,t+1) + obj.prms{14}*dI(:,:,t+1) + obj.prms{15}*dO(:,:,t)...
                    + dc.*obj.states{8}(:,:,t+1);
                
                dF(:,:,t) = dc.*obj.states{3}(:,:,t) .* obj.dsigmoid(obj.states{5}(:,:,t));
                dI(:,:,t) = dc.*obj.states{2}(:,:,t) .* obj.dsigmoid(obj.states{6}(:,:,t));
                dz(:,:,t) = dc.*obj.states{9}(:,:,t) .* obj.dtanh(obj.states{1}(:,:,t));
                
                gradR_z_tmp = gradR_z_tmp + dz(:,:,t)*obj.states{4}(:,:,t)';
                gradR_f_tmp = gradR_f_tmp + dF(:,:,t)*obj.states{4}(:,:,t)';
                gradR_i_tmp = gradR_i_tmp + dI(:,:,t)*obj.states{4}(:,:,t)';
                gradR_o_tmp = gradR_o_tmp + dO(:,:,t)*obj.states{4}(:,:,t)';
                
                gradP_f_tmp = gradP_f_tmp + dF(:,:,t).*obj.states{3}(:,:,t);
                gradP_i_tmp = gradP_i_tmp + dI(:,:,t).*obj.states{3}(:,:,t);
                gradP_o_tmp = gradP_o_tmp + dO(:,:,t).*obj.states{3}(:,:,t+1);
            end
            
            obj.gprms{5} = gradR_z_tmp./obj.batchSize;
            obj.gprms{6} = gradR_f_tmp./obj.batchSize;
            obj.gprms{7} = gradR_i_tmp./obj.batchSize;
            obj.gprms{8} = gradR_o_tmp./obj.batchSize;
            
            obj.gprms{13} = diag(mean(gradP_f_tmp, 2));
            obj.gprms{14} = diag(mean(gradP_i_tmp, 2));
            obj.gprms{15} = diag(mean(gradP_o_tmp, 2));
            
            dgate = {dz, dF, dI, dO};
        end
        
        function delta = bpropDelta(obj, dgate)
            dz = dgate{1};
            dF = dgate{2};
            dI = dgate{3};
            dO = dgate{4};
            
            gradW_z_tmp = 0; gradW_o_tmp = 0; gradW_f_tmp = 0; gradW_i_tmp = 0;
            gradb_z_tmp = 0; gradb_o_tmp = 0; gradb_f_tmp = 0; gradb_i_tmp = 0;
            
            for t=obj.T:-1:1
                obj.delta(:,:,t) = obj.prms{1}'*dz(:,:,t) + obj.prms{2}'*dF(:,:,t) + obj.prms{3}'*dI(:,:,t) + obj.prms{4}'*dO(:,:,t);

                gradW_z_tmp = gradW_z_tmp + dz(:,:,t)*obj.input(:,:,t)';
                gradW_f_tmp = gradW_f_tmp + dF(:,:,t)*obj.input(:,:,t)';
                gradW_i_tmp = gradW_i_tmp + dI(:,:,t)*obj.input(:,:,t)';
                gradW_o_tmp = gradW_o_tmp + dO(:,:,t)*obj.input(:,:,t)';

                gradb_z_tmp = gradb_z_tmp + dz(:,:,t);
                gradb_f_tmp = gradb_f_tmp + dF(:,:,t);
                gradb_i_tmp = gradb_i_tmp + dI(:,:,t);
                gradb_o_tmp = gradb_o_tmp + dO(:,:,t);
            end
            
            obj.gprms{1} = gradW_z_tmp./obj.batchSize;
            obj.gprms{2} = gradW_f_tmp./obj.batchSize;
            obj.gprms{3} = gradW_i_tmp./obj.batchSize;
            obj.gprms{4} = gradW_o_tmp./obj.batchSize;

            obj.gprms{9} = mean(gradb_z_tmp, 2);
            obj.gprms{10} = mean(gradb_f_tmp, 2);
            obj.gprms{11} = mean(gradb_i_tmp, 2);
            obj.gprms{12} = mean(gradb_o_tmp, 2);
            
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
        end
    end
end