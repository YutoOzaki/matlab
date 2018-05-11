classdef lossfun < basenode
    properties
        input, prms, grad, optm
        w = [0.5 0.5];
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
            x_recon = input.x_recon;
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

            loss = loss + 0.5.*sum(log(2*pi.*zsig) + 1);
            loss = obj.w(1).*0.5.*sum((x - x_recon).^2) - obj.w(2).*loss;
            
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
            SAFEDIV = 1e-8;
            
            if isa(x, 'gpuArray')
                dgam = zeros(K, batchsize, 'single', 'gpuArray');
                deta_mu = zeros(J, K, batchsize, 'single', 'gpuArray');
                deta_sig = zeros(J, K, batchsize, 'single', 'gpuArray');
            else
                dgam = zeros(K, batchsize);
                deta_mu = zeros(J, K, batchsize);
                deta_sig = zeros(J, K, batchsize);
            end
            
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
            
            dLdPI = bsxfun(@rdivide, gam, PI);
            dLdPI = sum(dLdPI, 2);
            
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
            deta_mu = sum(deta_mu, 3);
            deta_sig = sum(deta_sig, 3);
            
            A = bsxfun(@plus, -2.*x, 2.*xmu);
            dxmu = -1/(2*L) .* A./xsig;
            
            A = bsxfun(@minus, x, xmu).^2;
            dxsig = -1/(2*L) .* (1./xsig - A./(xsig.^2));
            
            A = 2.*bsxfun(@minus, zmu, repmat(reshape(eta_mu, [J, 1, K]), [1, batchsize, 1]));
            B = bsxfun(@rdivide, A, reshape(eta_sig, [J,1,K]));
            C = bsxfun(@times, B, reshape(gam',[1,batchsize,K]));
            dzmu = -0.5.*squeeze(sum(C,3));
            
            A = bsxfun(@rdivide, repmat(reshape(gam, [1,K,batchsize]), [J,1,1]), reshape(eta_sig, [J,K,1]));
            dzsig = -0.5.*squeeze(sum(A,2)) + 1./(2.*zsig);
            
            dLdx = -(x - obj.input.x_recon);
            
            delta = struct(...
                'gam', -obj.w(2).*dgam,...
                'PI', -obj.w(2).*dLdPI,...
                'eta_mu', -obj.w(2).*deta_mu,...
                'eta_sig', -obj.w(2).*deta_sig,...
                'x_recon', obj.w(1).*dLdx,...
                'xmu', -obj.w(2).*dxmu,...
                'xsig', -obj.w(2).*dxsig,...
                'zmu', -obj.w(2).*dzmu,...
                'zsig', -obj.w(2).*dzsig...
                );
            
            obj.grad = delta;
        end
        
        function init(obj)
        end
        
        function update(obj)
        end
        
        function refresh(obj)
        end
        
        function localgradcheck(obj)
            eps = 1e-6;
            f = zeros(2, 1);
            d = zeros(2, 1);
            
            batchsize = size(obj.input.x, 2);
            names = fieldnames(obj.grad);
            
            fprintf('---gradient checking (VDE loss function)---\n');
            for i=1:length(names)
                prm = obj.input.(names{i});
                
                lx = size(prm, 1);
                ly = size(prm, 2);

                ix = randi(lx);
                iy = randi(ly);
                val = prm(ix, iy);

                prm(ix, iy) = val + eps;
                obj.input.(names{i}) = prm;
                f(1) = obj.forwardprop(obj.input);
                
                prm(ix, iy) = val - eps;
                obj.input.(names{i}) = prm;
                f(2) = obj.forwardprop(obj.input);
                
                prm(ix, iy) = val;
                obj.input.(names{i}) = prm;

                d(1) = (f(1) - f(2))./(2*eps);
                d(2) = obj.grad.(names{i})(ix, iy, 1)./batchsize;
                
                re = abs(d(1) - d(2))./max(abs(d(1)), abs(d(2)));
                fprintf('%s: %e, %e, %e, %e\n', names{i}, val, d(1), d(2), re);
            end
        end
    end
end