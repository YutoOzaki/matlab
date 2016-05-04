classdef BLSTM < BaseLayer
   properties
        vis, hid, T, batchSize
        prms, states, gprms
        input, delta
    end
    
    properties (Constant)
        prmNum = 38;
        stateNum = 20;
        normInd = [1;5;6;7;11;15;16;17];
    end
    
    methods
        function initLayer(obj, vis, hid, T, batchSize, isBN)
            initLayer@BaseLayer(obj, vis, hid, T, batchSize, isBN);
            obj.delta = {obj.delta obj.delta};
        end
        
        function affineTrans(obj, x)
            if iscell(x) == false
                x = {x x.*0};
            end
            
            obj.input = x;
            
            b_zMat = repmat(obj.prms{9}, 1, obj.batchSize);
            b_fMat = repmat(obj.prms{10}, 1, obj.batchSize);
            b_iMat = repmat(obj.prms{11}, 1, obj.batchSize);
            b_oMat = repmat(obj.prms{12}, 1, obj.batchSize);
            
            for t=1:obj.T
                obj.states{1}(:,:,t) = obj.prms{1}*x{1}(:,:,t) + obj.prms{31}*x{2}(:,:,t) + b_zMat;
                obj.states{5}(:,:,t) = obj.prms{2}*x{1}(:,:,t) + obj.prms{32}*x{2}(:,:,t) + b_fMat;
                obj.states{6}(:,:,t) = obj.prms{3}*x{1}(:,:,t) + obj.prms{33}*x{2}(:,:,t) + b_iMat;
                obj.states{7}(:,:,t) = obj.prms{4}*x{1}(:,:,t) + obj.prms{34}*x{2}(:,:,t) + b_oMat;
            end
            
            b_zMat = repmat(obj.prms{24}, 1, obj.batchSize);
            b_fMat = repmat(obj.prms{25}, 1, obj.batchSize);
            b_iMat = repmat(obj.prms{26}, 1, obj.batchSize);
            b_oMat = repmat(obj.prms{27}, 1, obj.batchSize);
            
            for t=obj.T:-1:1
                obj.states{11}(:,:,t) = obj.prms{16}*x{2}(:,:,t) + obj.prms{35}*x{1}(:,:,t) + b_zMat;
                obj.states{15}(:,:,t) = obj.prms{17}*x{2}(:,:,t) + obj.prms{36}*x{1}(:,:,t) + b_fMat;
                obj.states{16}(:,:,t) = obj.prms{18}*x{2}(:,:,t) + obj.prms{37}*x{1}(:,:,t) + b_iMat;
                obj.states{17}(:,:,t) = obj.prms{19}*x{2}(:,:,t) + obj.prms{38}*x{1}(:,:,t) + b_oMat;
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
            
            for t=obj.T:-1:1
                obj.states{11}(:,:,t) = obj.states{11}(:,:,t) + obj.prms{20}*obj.states{14}(:,:,t+1);
                obj.states{12}(:,:,t) = tanh(obj.states{11}(:,:,t));

                obj.states{15}(:,:,t) = obj.states{15}(:,:,t) + obj.prms{21}*obj.states{14}(:,:,t+1) + obj.prms{28}*obj.states{13}(:,:,t+1);
                obj.states{18}(:,:,t+1) = obj.sigmoid(obj.states{15}(:,:,t));
                
                obj.states{16}(:,:,t) = obj.states{16}(:,:,t) + obj.prms{22}*obj.states{14}(:,:,t+1) + obj.prms{29}*obj.states{13}(:,:,t+1);
                obj.states{19}(:,:,t) = obj.sigmoid(obj.states{16}(:,:,t));

                obj.states{13}(:,:,t) = obj.states{12}(:,:,t).*obj.states{19}(:,:,t) + obj.states{13}(:,:,t+1).* obj.states{18}(:,:,t+1);
                
                obj.states{17}(:,:,t) = obj.states{17}(:,:,t) + obj.prms{23}*obj.states{14}(:,:,t+1) + obj.prms{30}*obj.states{13}(:,:,t);
                obj.states{20}(:,:,t) = obj.sigmoid(obj.states{17}(:,:,t));
                
                obj.states{14}(:,:,t) = tanh(obj.states{13}(:,:,t)).*obj.states{20}(:,:,t);
            end
            
            output = {obj.states{4}(:,:,2:end) obj.states{14}(:,:,1:obj.T)};
        end
        
        function dgate = bpropGate(obj, d)
            dgate = cell(length(obj.normInd),1);
            
            dz = repmat(obj.states{1}(:,:,1).*0, 1, 1, obj.T+1);
            dF = repmat(obj.states{5}(:,:,1).*0, 1, 1, obj.T+1);
            dI = repmat(obj.states{6}(:,:,1).*0, 1, 1, obj.T+1);
            dO = repmat(obj.states{7}(:,:,1).*0, 1, 1, obj.T+1);
            
            dc = obj.states{3}(:,:,1).*0;
            
            gradR_z_tmp = 0; gradR_o_tmp = 0; gradR_f_tmp = 0; gradR_i_tmp = 0;
            gradP_o_tmp = 0; gradP_f_tmp = 0; gradP_i_tmp = 0;
            
            for t=obj.T:-1:1
                dh = d{1}(:,:,t) + obj.prms{5}'*dz(:,:,t+1) + obj.prms{6}'*dF(:,:,t+1) + obj.prms{7}'*dI(:,:,t+1) + obj.prms{8}'*dO(:,:,t+1);
                
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
            
            dgate{1} = dz;
            dgate{2} = dF;
            dgate{3} = dI;
            dgate{4} = dO;
            
            dz = dz.*0; dI = dI.*0; dF = dF.*0;
            dO = dO.*0; dc = dc.*0;
            
            gradR_z_tmp = gradR_z_tmp.*0; gradR_o_tmp = gradR_o_tmp.*0; gradR_f_tmp = gradR_f_tmp.*0; gradR_i_tmp = gradR_i_tmp.*0;
            gradP_o_tmp = gradP_o_tmp.*0; gradP_f_tmp = gradP_f_tmp.*0; gradP_i_tmp = gradP_i_tmp.*0;
            
            for t=1:obj.T
                dh = d{2}(:,:,t) + obj.prms{20}'*dz(:,:,t) + obj.prms{21}'*dF(:,:,t) + obj.prms{22}'*dI(:,:,t) + obj.prms{23}'*dO(:,:,t);
                
                dO(:,:,t+1) = dh .* tanh(obj.states{13}(:,:,t)) .* obj.dsigmoid(obj.states{17}(:,:,t));
                
                dc = dh.*obj.states{20}(:,:,t).*obj.dtanh(obj.states{13}(:,:,t))...
                    + obj.prms{28}*dF(:,:,t) + obj.prms{29}*dI(:,:,t) + obj.prms{30}*dO(:,:,t+1)...
                    + dc.*obj.states{18}(:,:,t);
                
                dF(:,:,t+1) = dc.*obj.states{13}(:,:,t+1) .* obj.dsigmoid(obj.states{15}(:,:,t));
                dI(:,:,t+1) = dc.*obj.states{12}(:,:,t) .* obj.dsigmoid(obj.states{16}(:,:,t));
                dz(:,:,t+1) = dc.*obj.states{19}(:,:,t) .* obj.dtanh(obj.states{11}(:,:,t));
                
                gradR_z_tmp = gradR_z_tmp + dz(:,:,t+1)*obj.states{14}(:,:,t+1)';
                gradR_f_tmp = gradR_f_tmp + dF(:,:,t+1)*obj.states{14}(:,:,t+1)';
                gradR_i_tmp = gradR_i_tmp + dI(:,:,t+1)*obj.states{14}(:,:,t+1)';
                gradR_o_tmp = gradR_o_tmp + dO(:,:,t+1)*obj.states{14}(:,:,t+1)';

                gradP_f_tmp = gradP_f_tmp + dF(:,:,t+1).*obj.states{13}(:,:,t+1);
                gradP_i_tmp = gradP_i_tmp + dI(:,:,t+1).*obj.states{13}(:,:,t+1);
                gradP_o_tmp = gradP_o_tmp + dO(:,:,t+1).*obj.states{13}(:,:,t);
            end
            
            obj.gprms{20} = gradR_z_tmp./obj.batchSize;
            obj.gprms{21} = gradR_f_tmp./obj.batchSize;
            obj.gprms{22} = gradR_i_tmp./obj.batchSize;
            obj.gprms{23} = gradR_o_tmp./obj.batchSize;
            
            obj.gprms{28} = diag(mean(gradP_f_tmp, 2));
            obj.gprms{29} = diag(mean(gradP_i_tmp, 2));
            obj.gprms{30} = diag(mean(gradP_o_tmp, 2));
            
            dgate{5} = dz(:, :, 2:obj.T+1);
            dgate{6} = dF(:, :, 2:obj.T+1);
            dgate{7} = dI(:, :, 2:obj.T+1);
            dgate{8} = dO(:, :, 2:obj.T+1);
        end
        
        function delta = bpropDelta(obj, dgate)
            dz = dgate{1};
            dF = dgate{2};
            dI = dgate{3};
            dO = dgate{4};
            
            gradW_z_tmp = 0; gradW_o_tmp = 0; gradW_f_tmp = 0; gradW_i_tmp = 0;
            gradb_z_tmp = 0; gradb_o_tmp = 0; gradb_f_tmp = 0; gradb_i_tmp = 0;
            gradB_z_tmp = 0; gradB_f_tmp = 0; gradB_i_tmp = 0; gradB_o_tmp = 0;
            
            for t=obj.T:-1:1
                obj.delta{1}(:,:,t) = obj.prms{1}'*dz(:,:,t) + obj.prms{2}'*dF(:,:,t) + obj.prms{3}'*dI(:,:,t) + obj.prms{4}'*dO(:,:,t);
                obj.delta{2}(:,:,t) = obj.prms{31}'*dz(:,:,t) + obj.prms{32}'*dF(:,:,t) + obj.prms{33}'*dI(:,:,t) + obj.prms{34}'*dO(:,:,t);
                
                gradW_z_tmp = gradW_z_tmp + dz(:,:,t)*obj.input{1}(:,:,t)';
                gradW_f_tmp = gradW_f_tmp + dF(:,:,t)*obj.input{1}(:,:,t)';
                gradW_i_tmp = gradW_i_tmp + dI(:,:,t)*obj.input{1}(:,:,t)';
                gradW_o_tmp = gradW_o_tmp + dO(:,:,t)*obj.input{1}(:,:,t)';
                
                gradB_z_tmp = gradB_z_tmp + dz(:,:,t)*obj.input{2}(:,:,t)';
                gradB_f_tmp = gradB_f_tmp + dF(:,:,t)*obj.input{2}(:,:,t)';
                gradB_i_tmp = gradB_i_tmp + dI(:,:,t)*obj.input{2}(:,:,t)';
                gradB_o_tmp = gradB_o_tmp + dO(:,:,t)*obj.input{2}(:,:,t)';

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
            
            obj.gprms{31} = gradB_z_tmp./obj.batchSize;
            obj.gprms{32} = gradB_f_tmp./obj.batchSize;
            obj.gprms{33} = gradB_i_tmp./obj.batchSize;
            obj.gprms{34} = gradB_o_tmp./obj.batchSize;

            dz = dgate{5};
            dF = dgate{6};
            dI = dgate{7};
            dO = dgate{8};
            
            gradW_z_tmp = gradW_z_tmp.*0; gradW_o_tmp = gradW_o_tmp.*0; gradW_f_tmp = gradW_f_tmp.*0; gradW_i_tmp = gradW_i_tmp.*0;
            gradb_z_tmp = gradb_z_tmp.*0; gradb_o_tmp = gradb_o_tmp.*0; gradb_f_tmp = gradb_f_tmp.*0; gradb_i_tmp = gradb_i_tmp.*0;
            gradB_z_tmp = gradB_z_tmp.*0; gradB_f_tmp = gradB_f_tmp.*0; gradB_i_tmp = gradB_i_tmp.*0; gradB_o_tmp = gradB_o_tmp.*0;

            for t=1:obj.T
                obj.delta{2}(:,:,t) = obj.delta{2}(:,:,t) + obj.prms{16}'*dz(:,:,t) + obj.prms{17}'*dF(:,:,t) + obj.prms{18}'*dI(:,:,t) + obj.prms{19}'*dO(:,:,t);
                obj.delta{1}(:,:,t) = obj.delta{1}(:,:,t) + obj.prms{35}'*dz(:,:,t) + obj.prms{36}'*dF(:,:,t) + obj.prms{37}'*dI(:,:,t) + obj.prms{38}'*dO(:,:,t);
                
                gradW_z_tmp = gradW_z_tmp + dz(:,:,t)*obj.input{2}(:,:,t)';
                gradW_f_tmp = gradW_f_tmp + dF(:,:,t)*obj.input{2}(:,:,t)';
                gradW_i_tmp = gradW_i_tmp + dI(:,:,t)*obj.input{2}(:,:,t)';
                gradW_o_tmp = gradW_o_tmp + dO(:,:,t)*obj.input{2}(:,:,t)';
                
                gradB_z_tmp = gradB_z_tmp + dz(:,:,t)*obj.input{1}(:,:,t)';
                gradB_f_tmp = gradB_f_tmp + dF(:,:,t)*obj.input{1}(:,:,t)';
                gradB_i_tmp = gradB_i_tmp + dI(:,:,t)*obj.input{1}(:,:,t)';
                gradB_o_tmp = gradB_o_tmp + dO(:,:,t)*obj.input{1}(:,:,t)';
                
                gradb_z_tmp = gradb_z_tmp + dz(:,:,t);
                gradb_f_tmp = gradb_f_tmp + dF(:,:,t);
                gradb_i_tmp = gradb_i_tmp + dI(:,:,t);
                gradb_o_tmp = gradb_o_tmp + dO(:,:,t);
            end
            
            obj.gprms{16} = gradW_z_tmp./obj.batchSize;
            obj.gprms{17} = gradW_f_tmp./obj.batchSize;
            obj.gprms{18} = gradW_i_tmp./obj.batchSize;
            obj.gprms{19} = gradW_o_tmp./obj.batchSize;

            obj.gprms{24} = mean(gradb_z_tmp, 2);
            obj.gprms{25} = mean(gradb_f_tmp, 2);
            obj.gprms{26} = mean(gradb_i_tmp, 2);
            obj.gprms{27} = mean(gradb_o_tmp, 2);
            
            obj.gprms{35} = gradB_z_tmp./obj.batchSize;
            obj.gprms{36} = gradB_f_tmp./obj.batchSize;
            obj.gprms{37} = gradB_i_tmp./obj.batchSize;
            obj.gprms{38} = gradB_o_tmp./obj.batchSize;
            
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
            
            obj.prms{16} = 2.*(rand(obj.hid, obj.vis) - 0.5) .* sqrt(6/(obj.vis+obj.hid));
            obj.prms{17} = 2.*(rand(obj.hid, obj.vis) - 0.5) .* sqrt(6/(obj.vis+obj.hid));
            obj.prms{18} = 2.*(rand(obj.hid, obj.vis) - 0.5) .* sqrt(6/(obj.vis+obj.hid));
            obj.prms{19} = 2.*(rand(obj.hid, obj.vis) - 0.5) .* sqrt(6/(obj.vis+obj.hid));
            
            obj.prms{20} = 2.*(rand(obj.hid, obj.hid) - 0.5) .* sqrt(6/(obj.hid+obj.hid));
            obj.prms{21} = 2.*(rand(obj.hid, obj.hid) - 0.5) .* sqrt(6/(obj.hid+obj.hid));
            obj.prms{22} = 2.*(rand(obj.hid, obj.hid) - 0.5) .* sqrt(6/(obj.hid+obj.hid));
            obj.prms{23} = 2.*(rand(obj.hid, obj.hid) - 0.5) .* sqrt(6/(obj.hid+obj.hid));
            
            obj.prms{24} = zeros(obj.hid, 1);
            obj.prms{25} = zeros(obj.hid, 1);
            obj.prms{26} = zeros(obj.hid, 1);
            obj.prms{27} = zeros(obj.hid, 1);
            
            obj.prms{28} = diag(2.*(rand(obj.hid, 1) - 0.5) .* sqrt(6/(obj.hid+1)));
            obj.prms{29} = diag(2.*(rand(obj.hid, 1) - 0.5) .* sqrt(6/(obj.hid+1)));
            obj.prms{30} = diag(2.*(rand(obj.hid, 1) - 0.5) .* sqrt(6/(obj.hid+1)));
            
            obj.prms{31} = 2.*(rand(obj.hid, obj.vis) - 0.5) .* sqrt(6/(obj.vis+obj.hid));
            obj.prms{32} = 2.*(rand(obj.hid, obj.vis) - 0.5) .* sqrt(6/(obj.vis+obj.hid));
            obj.prms{33} = 2.*(rand(obj.hid, obj.vis) - 0.5) .* sqrt(6/(obj.vis+obj.hid));
            obj.prms{34} = 2.*(rand(obj.hid, obj.vis) - 0.5) .* sqrt(6/(obj.vis+obj.hid));
            obj.prms{35} = 2.*(rand(obj.hid, obj.vis) - 0.5) .* sqrt(6/(obj.vis+obj.hid));
            obj.prms{36} = 2.*(rand(obj.hid, obj.vis) - 0.5) .* sqrt(6/(obj.vis+obj.hid));
            obj.prms{37} = 2.*(rand(obj.hid, obj.vis) - 0.5) .* sqrt(6/(obj.vis+obj.hid));
            obj.prms{38} = 2.*(rand(obj.hid, obj.vis) - 0.5) .* sqrt(6/(obj.vis+obj.hid));
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