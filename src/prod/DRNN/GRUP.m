classdef GRUP < BaseLayer
    properties
        vis, hid, T, batchSize
        prms, states, gprms
        input, delta
    end
    
    properties (Constant)
        prmNum = 10;
        stateNum = 8;
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
                obj.states{1}(:,:,t) = obj.states{1}(:,:,t) + obj.prms{4}*obj.states{8}(:,:,t);
                obj.states{2}(:,:,t) = obj.sigmoid(obj.states{1}(:,:,t));

                obj.states{3}(:,:,t) = obj.states{3}(:,:,t) + obj.prms{5}*obj.states{8}(:,:,t);
                obj.states{4}(:,:,t) = obj.sigmoid(obj.states{3}(:,:,t));
                
                obj.states{5}(:,:,t) =obj.states{5}(:,:,t) + obj.states{4}(:,:,t).*(obj.prms{6}*obj.states{8}(:,:,t));
                obj.states{6}(:,:,t) = tanh(obj.states{5}(:,:,t));

                obj.states{7}(:,:,t+1) = (1 - obj.states{2}(:,:,t)).*obj.states{8}(:,:,t) + obj.states{2}(:,:,t).*obj.states{6}(:,:,t);
                
                obj.states{8}(:,:,t+1) = obj.prms{10}*obj.states{7}(:,:,t+1);
            end
            
            output = obj.states{8}(:,:,2:end);
        end
        
        function dgate = bpropGate(obj, d)
            dz = repmat(obj.states{1}(:,:,1).*0, 1, 1, obj.T+1);
            dr = repmat(obj.states{3}(:,:,1).*0, 1, 1, obj.T+1);
            du = repmat(obj.states{5}(:,:,1).*0, 1, 1, obj.T+1);
            
            dh = obj.states{7}(:,:,1).*0;
            
            gradR_z = 0; gradR_r = 0; gradR_u = 0; gradR_p = 0;
            
            for t=obj.T:-1:1
                dh = d(:,:,t) + dh.*(1 - obj.states{2}(:,:,t+1))...
                    + obj.prms{4}'*dz(:,:,t+1)...
                    + obj.prms{5}'*dr(:,:,t+1)...
                    + obj.prms{6}'*(du(:,:,t+1).*obj.states{4}(:,:,t+1));
                
                gradR_p = gradR_p + dh*obj.states{7}(:,:,t+1)';
                
                dh = obj.prms{10}'*dh;

                du(:,:,t) = dh .* obj.states{2}(:,:,t) .* obj.dtanh(obj.states{5}(:,:,t));
                dz(:,:,t) = dh .* (-obj.states{8}(:,:,t) + obj.states{6}(:,:,t)) .* obj.dsigmoid(obj.states{1}(:,:,t));
                dr(:,:,t) = du(:,:,t) .* (obj.prms{6}*obj.states{8}(:,:,t)) .* obj.dsigmoid(obj.states{3}(:,:,t));
                
                gradR_z = gradR_z + dz(:,:,t)*obj.states{8}(:,:,t)';
                gradR_r = gradR_r + dr(:,:,t)*obj.states{8}(:,:,t)';
                gradR_u = gradR_u + (du(:,:,t).*obj.states{4}(:,:,t))*obj.states{8}(:,:,t)';
            end
            
            obj.gprms{4} = gradR_z./obj.batchSize;
            obj.gprms{5} = gradR_r./obj.batchSize;
            obj.gprms{6} = gradR_u./obj.batchSize;
            
            obj.gprms{10} = gradR_p./obj.batchSize;
            
            dgate = {dz, dr, du};
        end
        
        function delta = bpropDelta(obj, dgate)
            dz = dgate{1};
            dr = dgate{2};
            du = dgate{3};
            
            gradW_z = 0;  gradb_z = 0;
            gradW_r = 0;  gradb_r = 0;
            gradW_u = 0;  gradb_u = 0;
            
            for t=obj.T:-1:1
                gradW_z = gradW_z + dz(:,:,t)*obj.input(:,:,t)';
                gradW_r = gradW_r + dr(:,:,t)*obj.input(:,:,t)';
                gradW_u = gradW_u + du(:,:,t)*obj.input(:,:,t)';

                gradb_z = gradb_z + dz(:,:,t);
                gradb_r = gradb_r + dr(:,:,t);
                gradb_u = gradb_u + du(:,:,t);
                
                obj.delta(:,:,t) = obj.prms{1}'*dz(:,:,t) + obj.prms{2}'*dr(:,:,t) + obj.prms{3}'*du(:,:,t);
            end
            
            obj.gprms{1} = gradW_z./obj.batchSize;
            obj.gprms{2} = gradW_r./obj.batchSize;
            obj.gprms{3} = gradW_u./obj.batchSize;

            obj.gprms{7} = mean(gradb_z, 2);
            obj.gprms{8} = mean(gradb_r, 2);
            obj.gprms{9} = mean(gradb_u, 2);
            
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
            
            obj.prms{10} = 2.*(rand(obj.hid, obj.hid) - 0.5) .* sqrt(6/(obj.hid+obj.hid));
        end
        
        function initStates(obj)
            obj.states{1} = zeros(obj.hid, obj.batchSize, obj.T);   % _z
            obj.states{2} = zeros(obj.hid, obj.batchSize, obj.T+1); % z
            obj.states{3} = zeros(obj.hid, obj.batchSize, obj.T);   % _r
            obj.states{4} = zeros(obj.hid, obj.batchSize, obj.T+1); % r
            obj.states{5} = zeros(obj.hid, obj.batchSize, obj.T);   % _u
            obj.states{6} = zeros(obj.hid, obj.batchSize, obj.T);   % u
            obj.states{7} = zeros(obj.hid, obj.batchSize, obj.T+1); % h
            obj.states{8} = zeros(obj.hid, obj.batchSize, obj.T+1); % p (projection)
        end
    end
end