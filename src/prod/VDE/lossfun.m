classdef lossfun < basenode
    properties
        input, prms, grad, optm
    end
    
    methods
        function obj = lossfun()
            obj.prms = [];
            obj.optm = [];
        end
        
        function output = forwardprop(obj, input)
            obj.input = input;
            loss = 0;
            
            x = input.x;
            xmu = input.xmu;
            xsig = input.xsig;
            zmu = input.zmu;
            zsig = input.zsig;
            gam = input.gam;
            eta_mu = input.eta_mu;
            eta_sig = input.eta_sig;
            PI = input.PI;
    
            [~, batchsize] = size(x);
            [~, ~, L] = size(xsig);
            K = length(PI);

            buf = 0;
            for l=1:L
                buf = buf + sum(log(2*pi.*xsig(:,:,l)) + (x - xmu(:,:,l)).^2./xsig(:,:,l));
            end
            loss = loss - buf./(2*L);

            buf = 0;
            for k=1:K
                A = bsxfun(@minus, zmu, eta_mu(:, k)).^2;
                B = bsxfun(@rdivide, A, eta_sig(:, k));
                C = bsxfun(@rdivide, zsig, eta_sig(:, k)) + B;
                D = bsxfun(@plus, log(2*pi.*eta_sig(:, k)), C);

                buf = buf + gam(k, :).*sum(D);
            end
            loss = loss - 0.5.*buf;

            A = bsxfun(@rdivide, PI, gam);
            idx = A == Inf;
            A(idx) = 1;
            loss = loss + sum(gam.*log(A));

            loss = loss + 0.5.*sum(log(2*pi.*zsig + 1));

            output = sum(loss)/batchsize;
        end
        
        function delta = backwardprop(obj, input)
            x = obj.input.x;
            xmu = obj.input.xmu;
            xsig = obj.input.xsig;
            eta_mu = obj.input.eta_mu;
            eta_sig = obj.input.eta_sig;
            gam = obj.input.gam;
            zmu = obj.input.zmu;
            zsig = obj.input.zsig;
            PI = obj.input.PI;
            
            [~, batchsize] = size(x);
            [~, ~, L] = size(xsig);
            K = length(PI);
            J = size(zmu, 1);
            SAFEDIV = 1e-75; %11e-75 > 1e-90 > 1e-60 > 1e-30
            
            dgam = zeros(K, batchsize);
            for k=1:K
                buf = log(bsxfun(@rdivide, PI(k), gam(k,:)));
                idx = buf == Inf;
                buf(idx) = log(PI(k)/SAFEDIV);
                
                A = bsxfun(@minus, zmu, eta_mu(:, k)).^2;
                B = bsxfun(@rdivide, A, eta_sig(:, k));
                C = bsxfun(@rdivide, zsig, eta_sig(:, k)) + B;
                D = bsxfun(@plus, log(2*pi.*eta_sig(:, k)), C);
                
                dgam(k, :) = -0.5.*sum(D) + buf - 1;
            end
            
            dPI = bsxfun(@rdivide, gam, PI);
            
            deta_mu = zeros(J, K, batchsize);
            deta_sig = zeros(J, K, batchsize);
            for k=1:K
                A = 2.*bsxfun(@plus, -zmu, eta_mu(:, k));
                B = bsxfun(@rdivide, A, eta_sig(:, k));
                deta_mu(:, k, :) = -0.5 .* bsxfun(@times, B, gam(k, :));
                
                A = bsxfun(@minus, zmu, eta_mu(:, k)).^2;
                B = bsxfun(@rdivide, A, eta_sig(:, k).^2);
                C = bsxfun(@rdivide, zsig, eta_sig(:, k).^2);
                D = bsxfun(@minus, 1./eta_sig(:, k), B + C);
                
                deta_sig(:, k, :) = -0.5 .* bsxfun(@times, D, gam(k, :));
            end
            
            A = bsxfun(@plus, -2.*x, 2.*xmu);
            dxmu = -1/(2*L) .* A./xsig;
            
            A = bsxfun(@minus, x, xmu).^2;
            dxsig = -1/(2*L) .* (1./xsig - A./(xsig.^2));
            
            delta = struct(...
                'gam', dgam,...
                'PI', dPI,...
                'eta_mu', deta_mu,...
                'eta_sig', deta_sig,...
                'xmu', dxmu,...
                'xsig', dxsig...
                );
            
            obj.grad = delta;
        end
        
        function init(obj)
        end
        
        function update(obj)
        end
        
        function localgradcheck(obj)
            cache = obj.input;
            
            f = zeros(2, 1);
            d = zeros(2, 1);
            eps = 1e-5;
            
            input = obj.input;
            names = fieldnames(input);
            batchsize = size(obj.input.x, 2);
            names = {'eta_mu', 'eta_sig', 'gam', 'PI', 'xmu', 'xsig'};
            
            for i=1:length(names)
                prm = input.(names{i});
                
                lx = size(prm, 1);
                ly = size(prm, 2);

                ix = randi(lx);
                iy = randi(ly);
                val = prm(ix, iy);

                prm(ix, iy) = val + eps;
                input.(names{i}) = prm;
                f(1) = obj.forwardprop(input);
                
                prm(ix, iy) = val - eps;
                input.(names{i}) = prm;
                f(2) = obj.forwardprop(input);
                
                prm(ix, iy) = val;
                input.(names{i}) = prm;

                d(1) = (f(1) - f(2))./(2*eps);
                
                if strcmp(names{i}, 'PI')
                    d(2) = sum(obj.grad.(names{i})(ix, :))./batchsize;
                elseif strcmp(names{i}, 'gam')
                    d(2) = obj.grad.(names{i})(ix, iy)./batchsize;
                elseif strcmp(names{i}, 'eta_mu') || strcmp(names{i}, 'eta_sig')
                    d(2) = sum(obj.grad.(names{i})(ix, iy, :))./batchsize;
                elseif strcmp(names{i}, 'xmu') || strcmp(names{i}, 'xsig')
                    d(2) = obj.grad.(names{i})(ix, iy, 1)./batchsize;
                end
                
                re = abs(d(1) - d(2))./max(abs(d(1)), abs(d(2)));
                fprintf('%s: %e, %e, %e, %e\n', names{i}, val, d(1), d(2), re);
            end
            
            obj.input = cache;
        end
    end
end