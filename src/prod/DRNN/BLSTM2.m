classdef BLSTM2 < BaseLayer
   properties
        vis, hid, T, batchSize
        prms, states, gprms, updatePrms
        input, delta
        updateFun
    end
    
    properties (Constant)
        prmNum = 26;
        stateNum = 20;
    end
    
    methods
        function obj = BLSTM2(vis, hid, T, batchSize)
            initLayer(obj, vis, hid, T, batchSize);
        end
        
        function output = fprop(obj, x)            
            obj.input = x;
            
            b_zMat = repmat(obj.prms{9}, 1, obj.batchSize);
            b_fMat = repmat(obj.prms{10}, 1, obj.batchSize);
            b_iMat = repmat(obj.prms{11}, 1, obj.batchSize);
            b_oMat = repmat(obj.prms{12}, 1, obj.batchSize);
            
            % states: u, z, c, h, F, I, O, gF, gI, gO
            for t=1:obj.T
                obj.states{1}(:,:,t) = obj.prms{1}*x(:,:,t) + obj.prms{5}*obj.states{4}(:,:,t) + b_zMat;
                obj.states{2}(:,:,t) = tanh(obj.states{1}(:,:,t));

                obj.states{5}(:,:,t) = obj.prms{2}*x(:,:,t) + obj.prms{6}*obj.states{4}(:,:,t) + b_fMat + obj.prms{13}*obj.states{3}(:,:,t);
                obj.states{8}(:,:,t) = sigmoid(obj.states{5}(:,:,t));
                
                obj.states{6}(:,:,t) = obj.prms{3}*x(:,:,t) + obj.prms{7}*obj.states{4}(:,:,t) + b_iMat + obj.prms{14}*obj.states{3}(:,:,t);
                obj.states{9}(:,:,t) = sigmoid(obj.states{6}(:,:,t));

                obj.states{3}(:,:,t+1) = obj.states{2}(:,:,t).*obj.states{9}(:,:,t) + obj.states{3}(:,:,t).*obj.states{8}(:,:,t);
                
                obj.states{7}(:,:,t) = obj.prms{4}*x(:,:,t) + obj.prms{8}*obj.states{4}(:,:,t) + b_oMat + obj.prms{15}*obj.states{3}(:,:,t+1);
                obj.states{10}(:,:,t) = sigmoid(obj.states{7}(:,:,t));
                
                obj.states{4}(:,:,t+1) = tanh(obj.states{3}(:,:,t+1)) .* obj.states{10}(:,:,t);
            end
            
            b_zMat = repmat(obj.prms{20}, 1, obj.batchSize);
            b_fMat = repmat(obj.prms{21}, 1, obj.batchSize);
            b_iMat = repmat(obj.prms{22}, 1, obj.batchSize);
            b_oMat = repmat(obj.prms{23}, 1, obj.batchSize);
            
            for t=obj.T:-1:1
                obj.states{11}(:,:,t) = obj.prms{1}*x(:,:,t) + obj.prms{16}*obj.states{14}(:,:,t+1) + b_zMat;
                obj.states{12}(:,:,t) = tanh(obj.states{11}(:,:,t));

                obj.states{15}(:,:,t) = obj.prms{2}*x(:,:,t) + obj.prms{17}*obj.states{14}(:,:,t+1) + b_fMat + obj.prms{24}*obj.states{13}(:,:,t+1);
                obj.states{18}(:,:,t+1) = sigmoid(obj.states{15}(:,:,t));
                
                obj.states{16}(:,:,t) = obj.prms{3}*x(:,:,t) + obj.prms{18}*obj.states{14}(:,:,t+1) + b_iMat + obj.prms{25}*obj.states{13}(:,:,t+1);
                obj.states{19}(:,:,t) = sigmoid(obj.states{16}(:,:,t));

                obj.states{13}(:,:,t) = obj.states{12}(:,:,t).*obj.states{19}(:,:,t) + obj.states{13}(:,:,t+1).* obj.states{18}(:,:,t+1);
                
                obj.states{17}(:,:,t) = obj.prms{4}*x(:,:,t) + obj.prms{19}*obj.states{14}(:,:,t+1) + b_oMat + obj.prms{26}*obj.states{13}(:,:,t);
                obj.states{20}(:,:,t) = sigmoid(obj.states{17}(:,:,t));
                
                obj.states{14}(:,:,t) = tanh(obj.states{13}(:,:,t)).*obj.states{20}(:,:,t);
            end
            
            output = obj.states{4}(:,:,2:end) + obj.states{14}(:,:,1:obj.T);
        end
        
        function delta = bprop(obj,d)         
            dz = obj.states{1}(:,:,1).*0;
            dF = obj.states{5}(:,:,1).*0;
            dI = obj.states{6}(:,:,1).*0;
            dO = obj.states{7}(:,:,1).*0;
            dc = obj.states{3}(:,:,1).*0;
            
            gradW_z_tmp = 0; gradW_o_tmp = 0; gradW_f_tmp = 0; gradW_i_tmp = 0;
            gradR_z_tmp = 0; gradR_o_tmp = 0; gradR_f_tmp = 0; gradR_i_tmp = 0;
            gradb_z_tmp = 0; gradb_o_tmp = 0; gradb_f_tmp = 0; gradb_i_tmp = 0;
            gradP_o_tmp = 0; gradP_f_tmp = 0; gradP_i_tmp = 0;

            for t=obj.T:-1:1
                dh = d(:,:,t) + obj.prms{5}'*dz + obj.prms{6}'*dF + obj.prms{7}'*dI + obj.prms{8}'*dO;
                
                dO = dh .* tanh(obj.states{3}(:,:,t+1)) .* obj.dsigmoid(obj.states{7}(:,:,t));
                
                dc = dh.*obj.states{10}(:,:,t).*obj.dtanh(obj.states{3}(:,:,t+1))...
                    + obj.prms{13}*dF + obj.prms{14}*dI + obj.prms{15}*dO...
                    + dc.*obj.states{8}(:,:,t+1);
                
                dF = dc.*obj.states{3}(:,:,t) .* obj.dsigmoid(obj.states{5}(:,:,t));
                dI = dc.*obj.states{2}(:,:,t) .* obj.dsigmoid(obj.states{6}(:,:,t));
                dz = dc.*obj.states{9}(:,:,t) .* obj.dtanh(obj.states{1}(:,:,t));

                obj.delta(:,:,t) = obj.prms{1}'*dz + obj.prms{2}'*dF + obj.prms{3}'*dI + obj.prms{4}'*dO;
                
                gradR_z_tmp = gradR_z_tmp + dz*obj.states{4}(:,:,t)';
                gradR_f_tmp = gradR_f_tmp + dF*obj.states{4}(:,:,t)';
                gradR_i_tmp = gradR_i_tmp + dI*obj.states{4}(:,:,t)';
                gradR_o_tmp = gradR_o_tmp + dO*obj.states{4}(:,:,t)';

                gradP_f_tmp = gradP_f_tmp + dF.*obj.states{3}(:,:,t);
                gradP_i_tmp = gradP_i_tmp + dI.*obj.states{3}(:,:,t);
                gradP_o_tmp = gradP_o_tmp + dO.*obj.states{3}(:,:,t+1);

                gradW_z_tmp = gradW_z_tmp + dz*obj.input(:,:,t)';
                gradW_f_tmp = gradW_f_tmp + dF*obj.input(:,:,t)';
                gradW_i_tmp = gradW_i_tmp + dI*obj.input(:,:,t)';
                gradW_o_tmp = gradW_o_tmp + dO*obj.input(:,:,t)';

                gradb_z_tmp = gradb_z_tmp + dz;
                gradb_f_tmp = gradb_f_tmp + dF;
                gradb_i_tmp = gradb_i_tmp + dI;
                gradb_o_tmp = gradb_o_tmp + dO;
            end
            
            obj.gprms{1} = gradW_z_tmp./obj.batchSize;
            obj.gprms{2} = gradW_f_tmp./obj.batchSize;
            obj.gprms{3} = gradW_i_tmp./obj.batchSize;
            obj.gprms{4} = gradW_o_tmp./obj.batchSize;

            obj.gprms{5} = gradR_z_tmp./obj.batchSize;
            obj.gprms{6} = gradR_f_tmp./obj.batchSize;
            obj.gprms{7} = gradR_i_tmp./obj.batchSize;
            obj.gprms{8} = gradR_o_tmp./obj.batchSize;

            obj.gprms{9} = mean(gradb_z_tmp, 2);
            obj.gprms{10} = mean(gradb_f_tmp, 2);
            obj.gprms{11} = mean(gradb_i_tmp, 2);
            obj.gprms{12} = mean(gradb_o_tmp, 2);
            
            obj.gprms{13} = diag(mean(gradP_f_tmp, 2));
            obj.gprms{14} = diag(mean(gradP_i_tmp, 2));
            obj.gprms{15} = diag(mean(gradP_o_tmp, 2));

            dz = dz.*0; dI = dI.*0; dF = dF.*0;
            dO = dO.*0; dc = dc.*0;
            
            gradW_z_tmp = gradW_z_tmp.*0; gradW_o_tmp = gradW_o_tmp.*0; gradW_f_tmp = gradW_f_tmp.*0; gradW_i_tmp = gradW_i_tmp.*0;
            gradR_z_tmp = gradR_z_tmp.*0; gradR_o_tmp = gradR_o_tmp.*0; gradR_f_tmp = gradR_f_tmp.*0; gradR_i_tmp = gradR_i_tmp.*0;
            gradb_z_tmp = gradb_z_tmp.*0; gradb_o_tmp = gradb_o_tmp.*0; gradb_f_tmp = gradb_f_tmp.*0; gradb_i_tmp = gradb_i_tmp.*0;
            gradP_o_tmp = gradP_o_tmp.*0; gradP_f_tmp = gradP_f_tmp.*0; gradP_i_tmp = gradP_i_tmp.*0;

            for t=1:obj.T
                dh = d(:,:,t) + obj.prms{16}'*dz + obj.prms{17}'*dF + obj.prms{18}'*dI + obj.prms{19}'*dO;
                
                dO = dh .* tanh(obj.states{13}(:,:,t)) .* obj.dsigmoid(obj.states{17}(:,:,t));
                
                dc = dh.*obj.states{20}(:,:,t).*obj.dtanh(obj.states{13}(:,:,t))...
                    + obj.prms{24}*dF + obj.prms{25}*dI + obj.prms{26}*dO...
                    + dc.*obj.states{18}(:,:,t);
                
                dF = dc.*obj.states{13}(:,:,t+1) .* obj.dsigmoid(obj.states{15}(:,:,t));
                dI = dc.*obj.states{12}(:,:,t) .* obj.dsigmoid(obj.states{16}(:,:,t));
                dz = dc.*obj.states{19}(:,:,t) .* obj.dtanh(obj.states{11}(:,:,t));

                obj.delta(:,:,t) = obj.delta(:,:,t) + obj.prms{1}'*dz + obj.prms{2}'*dF + obj.prms{3}'*dI + obj.prms{4}'*dO;
                
                gradR_z_tmp = gradR_z_tmp + dz*obj.states{14}(:,:,t+1)';
                gradR_f_tmp = gradR_f_tmp + dF*obj.states{14}(:,:,t+1)';
                gradR_i_tmp = gradR_i_tmp + dI*obj.states{14}(:,:,t+1)';
                gradR_o_tmp = gradR_o_tmp + dO*obj.states{14}(:,:,t+1)';

                gradP_f_tmp = gradP_f_tmp + dF.*obj.states{13}(:,:,t+1);
                gradP_i_tmp = gradP_i_tmp + dI.*obj.states{13}(:,:,t+1);
                gradP_o_tmp = gradP_o_tmp + dO.*obj.states{13}(:,:,t);

                gradW_z_tmp = gradW_z_tmp + dz*obj.input(:,:,t)';
                gradW_f_tmp = gradW_f_tmp + dF*obj.input(:,:,t)';
                gradW_i_tmp = gradW_i_tmp + dI*obj.input(:,:,t)';
                gradW_o_tmp = gradW_o_tmp + dO*obj.input(:,:,t)';
                
                gradb_z_tmp = gradb_z_tmp + dz;
                gradb_f_tmp = gradb_f_tmp + dF;
                gradb_i_tmp = gradb_i_tmp + dI;
                gradb_o_tmp = gradb_o_tmp + dO;
            end
            
            obj.gprms{1} = obj.gprms{1} + gradW_z_tmp./obj.batchSize;
            obj.gprms{2} = obj.gprms{2} + gradW_f_tmp./obj.batchSize;
            obj.gprms{3} = obj.gprms{3} + gradW_i_tmp./obj.batchSize;
            obj.gprms{4} = obj.gprms{4} + gradW_o_tmp./obj.batchSize;

            obj.gprms{16} = gradR_z_tmp./obj.batchSize;
            obj.gprms{17} = gradR_f_tmp./obj.batchSize;
            obj.gprms{18} = gradR_i_tmp./obj.batchSize;
            obj.gprms{19} = gradR_o_tmp./obj.batchSize;

            obj.gprms{20} = mean(gradb_z_tmp, 2);
            obj.gprms{21} = mean(gradb_f_tmp, 2);
            obj.gprms{22} = mean(gradb_i_tmp, 2);
            obj.gprms{23} = mean(gradb_o_tmp, 2);
            
            obj.gprms{24} = diag(mean(gradP_f_tmp, 2));
            obj.gprms{25} = diag(mean(gradP_i_tmp, 2));
            obj.gprms{26} = diag(mean(gradP_o_tmp, 2));
            
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
            obj.prms{17} = 2.*(rand(obj.hid, obj.hid) - 0.5) .* sqrt(6/(obj.hid+obj.hid));
            obj.prms{18} = 2.*(rand(obj.hid, obj.hid) - 0.5) .* sqrt(6/(obj.hid+obj.hid));
            obj.prms{19} = 2.*(rand(obj.hid, obj.hid) - 0.5) .* sqrt(6/(obj.hid+obj.hid));
            
            obj.prms{20} = zeros(obj.hid, 1);
            obj.prms{21} = zeros(obj.hid, 1);
            obj.prms{22} = zeros(obj.hid, 1);
            obj.prms{23} = zeros(obj.hid, 1);
            
            obj.prms{24} = diag(2.*(rand(obj.hid, 1) - 0.5) .* sqrt(6/(obj.hid+1)));
            obj.prms{25} = diag(2.*(rand(obj.hid, 1) - 0.5) .* sqrt(6/(obj.hid+1)));
            obj.prms{26} = diag(2.*(rand(obj.hid, 1) - 0.5) .* sqrt(6/(obj.hid+1)));
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
            
            obj.states{11} = zeros(obj.hid, obj.batchSize, obj.T);  % u
            obj.states{12} = zeros(obj.hid, obj.batchSize, obj.T);  % z
            obj.states{13} = zeros(obj.hid, obj.batchSize, obj.T+1);% c
            obj.states{14} = zeros(obj.hid, obj.batchSize, obj.T+1);% h
            obj.states{15} = zeros(obj.hid, obj.batchSize, obj.T);  % F
            obj.states{16} = zeros(obj.hid, obj.batchSize, obj.T);  % I
            obj.states{17} = zeros(obj.hid, obj.batchSize, obj.T);  % O
            obj.states{18} = zeros(obj.hid, obj.batchSize, obj.T+1);% gF
            obj.states{19} = zeros(obj.hid, obj.batchSize, obj.T);  % gI
            obj.states{20} = zeros(obj.hid, obj.batchSize, obj.T);  % gO
        end
    end
end