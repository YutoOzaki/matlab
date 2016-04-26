classdef GRU < BaseLayer
    properties
        vis, hid, T, batchSize
        prms, states, gprms, updatePrms, BNPrms
        input, delta
        updateFun
    end
    
    properties (Constant)
        prmNum = 9;
        stateNum = 7;
        normInd = [1;3;5];
    end
    
    methods
        function affineTrans(obj, x)
            obj.input = x;
            
            b_zMat = repmat(obj.prms{7}, 1, obj.batchSize);
            b_rMat = repmat(obj.prms{8}, 1, obj.batchSize);
            b_hMat = repmat(obj.prms{9}, 1, obj.batchSize);
            
            for t=1:obj.T
                obj.states{1}(:,:,t) = obj.prms{1}*x(:,:,t) + b_zMat;
                obj.states{3}(:,:,t) = obj.prms{2}*x(:,:,t) + b_rMat;
                obj.states{5}(:,:,t) = obj.prms{3}*x(:,:,t) + b_hMat;
            end
        end
        
        function output = nonlinearTrans(obj)
            % states: _z, z, _r, r, _u, u, h
            for t=1:obj.T
                obj.states{1}(:,:,t) = obj.states{1}(:,:,t) + obj.prms{4}*obj.states{7}(:,:,t);
                obj.states{2}(:,:,t) = sigmoid(obj.states{1}(:,:,t));

                obj.states{3}(:,:,t) = obj.states{3}(:,:,t) + obj.prms{5}*obj.states{7}(:,:,t);
                obj.states{4}(:,:,t) = sigmoid(obj.states{3}(:,:,t));
                
                obj.states{5}(:,:,t) =obj.states{5}(:,:,t) + obj.states{4}(:,:,t).*(obj.prms{6}*obj.states{7}(:,:,t));
                obj.states{6}(:,:,t) = tanh(obj.states{5}(:,:,t));

                obj.states{7}(:,:,t+1) = (1 - obj.states{2}(:,:,t)).*obj.states{7}(:,:,t) + obj.states{2}(:,:,t).*obj.states{6}(:,:,t);
            end
            
            output = obj.states{7}(:,:,2:end);
        end
        
        function delta = bprop(obj, d)
            dz = obj.states{1}(:,:,1).*0;
            dr = obj.states{3}(:,:,1).*0;
            du = obj.states{5}(:,:,1).*0;
            dh = obj.states{7}(:,:,1).*0;
            
            gradW_z_tmp = 0; gradW_r_tmp = 0; gradW_u_tmp = 0;
            gradR_z_tmp = 0; gradR_r_tmp = 0; gradR_u_tmp = 0;
            gradb_z_tmp = 0; gradb_r_tmp = 0; gradb_u_tmp = 0;

            for t=obj.T:-1:1
                dh = d(:,:,t) + dh.*(1 - obj.states{2}(:,:,t+1))...
                    + obj.prms{4}'*dz + obj.prms{5}'*dr + obj.prms{6}'*(du.*obj.states{4}(:,:,t+1));
                
                du = dh .* obj.states{2}(:,:,t) .* obj.dtanh(obj.states{5}(:,:,t));
                
                dz = dh .* (-obj.states{7}(:,:,t) + obj.states{6}(:,:,t)) .* obj.dsigmoid(obj.states{1}(:,:,t));
                
                dr = du .* (obj.prms{6}*obj.states{7}(:,:,t)) .* obj.dsigmoid(obj.states{3}(:,:,t));
                
                obj.delta(:,:,t) = obj.prms{1}'*dz + obj.prms{2}'*dr + obj.prms{3}'*du;

                gradR_z_tmp = gradR_z_tmp + dz*obj.states{7}(:,:,t)';
                gradR_r_tmp = gradR_r_tmp + dr*obj.states{7}(:,:,t)';
                gradR_u_tmp = gradR_u_tmp + (du.*obj.states{4}(:,:,t))*obj.states{7}(:,:,t)';

                gradW_z_tmp = gradW_z_tmp + dz*obj.input(:,:,t)';
                gradW_r_tmp = gradW_r_tmp + dr*obj.input(:,:,t)';
                gradW_u_tmp = gradW_u_tmp + du*obj.input(:,:,t)';

                gradb_z_tmp = gradb_z_tmp + dz;
                gradb_r_tmp = gradb_r_tmp + dr;
                gradb_u_tmp = gradb_u_tmp + du;
            end
            
            obj.gprms{1} = gradW_z_tmp./obj.batchSize;
            obj.gprms{2} = gradW_r_tmp./obj.batchSize;
            obj.gprms{3} = gradW_u_tmp./obj.batchSize;

            obj.gprms{4} = gradR_z_tmp./obj.batchSize;
            obj.gprms{5} = gradR_r_tmp./obj.batchSize;
            obj.gprms{6} = gradR_u_tmp./obj.batchSize;

            obj.gprms{7} = mean(gradb_z_tmp, 2);
            obj.gprms{8} = mean(gradb_r_tmp, 2);
            obj.gprms{9} = mean(gradb_u_tmp, 2);
            
            delta = obj.delta;
        end
        
        function initPrms(obj)
            obj.prms{1} = 2.*(rand(obj.hid, obj.vis) - 0.5) .* sqrt(6/(obj.vis+obj.hid));
            obj.prms{2} = 2.*(rand(obj.hid, obj.vis) - 0.5) .* sqrt(6/(obj.vis+obj.hid));
            obj.prms{3} = 2.*(rand(obj.hid, obj.vis) - 0.5) .* sqrt(6/(obj.vis+obj.hid));
            
            obj.prms{4} = 2.*(rand(obj.hid, obj.hid) - 0.5) .* sqrt(6/(obj.hid+obj.hid));
            obj.prms{5} = 2.*(rand(obj.hid, obj.hid) - 0.5) .* sqrt(6/(obj.hid+obj.hid));
            obj.prms{6} = 2.*(rand(obj.hid, obj.hid) - 0.5) .* sqrt(6/(obj.hid+obj.hid));
            
            obj.prms{7} = zeros(obj.hid, 1);
            obj.prms{8} = zeros(obj.hid, 1);
            obj.prms{9} = zeros(obj.hid, 1);
        end
        
        function initStates(obj)
            obj.states{1} = zeros(obj.hid, obj.batchSize, obj.T);   % _z
            obj.states{2} = zeros(obj.hid, obj.batchSize, obj.T+1); % z
            obj.states{3} = zeros(obj.hid, obj.batchSize, obj.T);   % _r
            obj.states{4} = zeros(obj.hid, obj.batchSize, obj.T+1); % r
            obj.states{5} = zeros(obj.hid, obj.batchSize, obj.T);   % _u
            obj.states{6} = zeros(obj.hid, obj.batchSize, obj.T);   % u
            obj.states{7} = zeros(obj.hid, obj.batchSize, obj.T+1); % h
        end
    end
    
    %{
    function delta = bprop(obj, d)
        dz = obj.states{1}(:,:,1).*0;
        dr = obj.states{3}(:,:,1).*0;
        du = obj.states{5}(:,:,1).*0;
        dh = obj.states{7}(:,:,1).*0;

        gradW_z_tmp = 0; gradW_r_tmp = 0; gradW_u_tmp = 0;
        gradR_z_tmp = 0; gradR_r_tmp = 0; gradR_u_tmp = 0;
        gradb_z_tmp = 0; gradb_r_tmp = 0; gradb_u_tmp = 0;

        for t=obj.T:-1:1
            dh = d(:,:,t) + dh.*(1 - obj.states{2}(:,:,t+1))...
                + obj.prms{4}'*dz + obj.prms{5}'*dr + obj.prms{6}'*(du.*obj.states{4}(:,:,t+1));

            du = dh .* obj.states{2}(:,:,t) .* obj.dtanh(obj.states{5}(:,:,t));

            dz = dh .* (-obj.states{7}(:,:,t) + obj.states{6}(:,:,t)) .* obj.dsigmoid(obj.states{1}(:,:,t));

            dr = du .* (obj.prms{6}*obj.states{7}(:,:,t)) .* obj.dsigmoid(obj.states{3}(:,:,t));

            obj.delta(:,:,t) = obj.prms{1}'*dz + obj.prms{2}'*dr + obj.prms{3}'*du;

            gradR_z_tmp = gradR_z_tmp + dz*obj.states{7}(:,:,t)';
            gradR_r_tmp = gradR_r_tmp + dr*obj.states{7}(:,:,t)';
            gradR_u_tmp = gradR_u_tmp + (du.*obj.states{4}(:,:,t))*obj.states{7}(:,:,t)';

            gradW_z_tmp = gradW_z_tmp + dz*obj.input(:,:,t)';
            gradW_r_tmp = gradW_r_tmp + dr*obj.input(:,:,t)';
            gradW_u_tmp = gradW_u_tmp + du*obj.input(:,:,t)';

            gradb_z_tmp = gradb_z_tmp + dz;
            gradb_r_tmp = gradb_r_tmp + dr;
            gradb_u_tmp = gradb_u_tmp + du;
        end

        obj.gprms{1} = gradW_z_tmp./obj.batchSize;
        obj.gprms{2} = gradW_r_tmp./obj.batchSize;
        obj.gprms{3} = gradW_u_tmp./obj.batchSize;

        obj.gprms{4} = gradR_z_tmp./obj.batchSize;
        obj.gprms{5} = gradR_r_tmp./obj.batchSize;
        obj.gprms{6} = gradR_u_tmp./obj.batchSize;

        obj.gprms{7} = mean(gradb_z_tmp, 2);
        obj.gprms{8} = mean(gradb_r_tmp, 2);
        obj.gprms{9} = mean(gradb_u_tmp, 2);

        delta = obj.delta;
    end
    %}
end