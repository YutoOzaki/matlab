classdef mogtrans < basenode
    properties
        input, prms, grad, optm
        K, J, gam
        gpuwrapper, mvnpdfwrapper
        
        SAFEDIV = 1e-12;
        DECAY = 1;
    end
    
    methods
        function obj = mogtrans(K, J, gam, optm, isdiag, gpumode)
            eta_mu = zeros(J, K);
            eta_lnsig = zeros(J, K);
            p = zeros(K, 1);
            
            if isdiag
                obj.mvnpdfwrapper = @diagmvnpdf;
            else
                obj.mvnpdfwrapper = @mvnpdf;
            end
            
            switch nargin
                case 6
                    if gpumode
                        eta_mu = gpuArray(cast(eta_mu, 'single'));
                        eta_lnsig = gpuArray(cast(eta_lnsig, 'single'));
                        p = gpuArray(cast(p, 'single'));
                        K = gpuArray(cast(K, 'single'));
                        J = gpuArray(cast(J, 'single'));
                        gam = gpuArray(cast(gam, 'single'));
                        
                        obj.gpuwrapper = @(x) gpuArray(cast(x, 'single'));
                    else
                        obj.gpuwrapper = @(x) x;
                    end
                otherwise
                    obj.gpuwrapper = @(x) x;
            end
            
            obj.K = K;
            obj.J = J;
            obj.gam = gam;
            
            obj.prms = struct(...
                'eta_mu', eta_mu,...
                'eta_lnsig', eta_lnsig,...
                'p', p...
                );
            
            obj.optm = optm;
        end
        
        function [output, r] = forwardprop(obj, input)
            PI = obj.getPI();
            eta_sig = exp(obj.prms.eta_lnsig);
            
            obj.input = input;
            [~, batchsize] = size(input);
            
            r = obj.gpuwrapper(zeros(obj.K, batchsize));
            
            for k=1:obj.K
                r(k, :) = PI(k) .* obj.mvnpdfwrapper(input', obj.prms.eta_mu(:, k)', diag(eta_sig(:, k)));
            end
            
            r = r + obj.SAFEDIV;
            
            output = bsxfun(@rdivide, r, sum(r));
        end
        
        function delta = backwardprop(obj, input)
            batchsize = size(obj.input, 2);
            
            dLdgam = input.gam;
            dLdPI = input.PI;
            dLdeta_mu = input.eta_mu;
            dLdeta_sig = input.eta_sig;
            
            eta_mu = obj.prms.eta_mu;
            eta_sig = exp(obj.prms.eta_lnsig);
            q = softmax(obj.prms.p);
            PI = obj.getPI();
            
            z_kpdf = obj.gpuwrapper(zeros(obj.K, batchsize));
            for k=1:obj.K
                z_kpdf(k, :) = obj.mvnpdfwrapper(obj.input', eta_mu(:, k)', diag(eta_sig(:, k)))';
            end
            
            dgamdPI = obj.gpuwrapper(zeros(obj.K, obj.K, batchsize));
            dgamdeta_mu = obj.gpuwrapper(zeros(obj.K, obj.K, obj.J, batchsize));
            dgamdeta_sig = obj.gpuwrapper(zeros(obj.K, obj.K, obj.J, batchsize));
            dgamdz = obj.gpuwrapper(zeros(obj.K, obj.J, batchsize));
            dPIdq = obj.gpuwrapper(zeros(obj.K, obj.K));
            dqdp = obj.gpuwrapper(zeros(obj.K, obj.K));
            geta_mu = obj.gpuwrapper(zeros(obj.J, obj.K)) + dLdeta_mu;
            geta_sig = obj.gpuwrapper(zeros(obj.J, obj.K)) + dLdeta_sig;
            
            I = sum(bsxfun(@times, PI, z_kpdf) + obj.SAFEDIV);
            M = bsxfun(@minus, reshape(obj.input, [obj.J,1,batchsize]), eta_mu);
            N = bsxfun(@rdivide, M, eta_sig);
            O = bsxfun(@times, reshape(-PI, [1,obj.K,1]), N);
            P = bsxfun(@times, reshape(z_kpdf, [1,obj.K,batchsize]), O);
            Q = squeeze(sum(P, 2));
            normalization = I.^2;
            
            idx = normalization == 0;
            normalization(idx) = 1e-12;
            
            for k=1:obj.K
                idx = setdiff(1:obj.K, k);
                buf = sum(bsxfun(@times, PI(idx), z_kpdf(idx, :)) + obj.SAFEDIV);
                
                dgamdPI(k, k, :) = z_kpdf(k, :).*buf./normalization;
                
                i = k;
                A = PI(i).*z_kpdf(i, :).*buf./normalization;
                
                B = bsxfun(@minus, obj.input, eta_mu(:, i));
                C = bsxfun(@rdivide, B, eta_sig(:, i));
                dgamdeta_mu(i, i, :, :) = bsxfun(@times, A, C);
                
                D = bsxfun(@plus, -eta_sig(:, k), B.^2);
                E = bsxfun(@rdivide, D, 2.*eta_sig(:, k).^2);
                dgamdeta_sig(k, k, :, :) = bsxfun(@times, A, E);
                
                D = bsxfun(@times, C, z_kpdf(i, :));
                R = bsxfun(@times, -PI(k).*D, I) - bsxfun(@times, PI(k).*z_kpdf(k, :), Q);
                dgamdz(k, :, :) = bsxfun(@rdivide, R, normalization);
                
                for i=1:(obj.K-1)
                    buf = -PI(k).*z_kpdf(k, :).*z_kpdf(idx(i), :)./normalization;
                    dgamdPI(k, idx(i), :) = buf;
                    
                    A = PI(idx(i)).*buf;
                    
                    B = bsxfun(@minus, obj.input, eta_mu(:, idx(i)));
                    C = bsxfun(@rdivide, B, eta_sig(:, idx(i)));
                    dgamdeta_mu(k, idx(i), :, :) = bsxfun(@times, A, C);
                    
                    C = bsxfun(@plus, -eta_sig(:, idx(i)), B.^2);
                    D = bsxfun(@rdivide, C, 2.*eta_sig(:, idx(i)).^2);
                    dgamdeta_sig(k, idx(i), :, :) = bsxfun(@times, A, D);
                end
            end
            
            dLdgam = reshape(dLdgam, [obj.K, 1, batchsize]);
            gPI = sum(sum(bsxfun(@times, dLdgam, dgamdPI), 1), 3)' + dLdPI;
            delta = squeeze(sum(bsxfun(@times, dLdgam, dgamdz), 1));
            
            geta_mu = geta_mu';
            geta_sig = geta_sig';
            for k=1:obj.K
                geta_mu(k, :) = geta_mu(k, :) + sum(sum(bsxfun(@times, squeeze(dgamdeta_mu(:, k, :, :)), dLdgam), 3), 1);
                geta_sig(k, :) = geta_sig(k, :) + sum(sum(bsxfun(@times, squeeze(dgamdeta_sig(:, k, :, :)), dLdgam), 3), 1);
            end
            geta_mu = geta_mu';
            geta_sig = geta_sig';
            
            geta_lnsig = geta_sig.*exp(obj.prms.eta_lnsig);
            
            for i=1:obj.K
                idx = setdiff(1:obj.K, i);
                
                dPIdq(i, i) = sum(q(idx))+ (obj.K - 1).*obj.gam;
                dqdp(i, i) =  exp(obj.prms.p(i))*sum(exp(obj.prms.p(idx)));
                
                dPIdq(i, idx) = -q(i) - obj.gam;
                dqdp(i, idx) = -exp(obj.prms.p(i) + obj.prms.p(idx));
            end
            dqdp = dqdp./sum(exp(obj.prms.p))^2;
            dPIdq = dPIdq./(sum(q) + obj.K*obj.gam)^2;
            gp = (gPI'*dPIdq*dqdp)';
            
            gp = gp./batchsize;
            geta_mu = geta_mu./batchsize;
            geta_lnsig = geta_lnsig./batchsize;
            
            obj.grad = struct(...
                'p', gp,...
                'eta_mu', geta_mu,...
                'eta_lnsig', geta_lnsig...
                );
        end
        
        function init(obj)
            [J, K] = size(obj.prms.eta_mu);
            assert(J == obj.J && K == obj.K, 'matrix size of parameters are wrong');
            
            obj.prms.p = obj.gpuwrapper(rand(K, 1));
            
            for k=1:obj.K
                obj.prms.eta_mu(:, k) = obj.gpuwrapper(rand(J, 1));
                obj.prms.eta_lnsig(:, k) = obj.gpuwrapper(log(rand(J, 1).^2));
            end
            
            obj.optm.init(obj.prms);
        end
        
        function update(obj)
            prmnames = fieldnames(obj.grad);
            
            for l=1:length(prmnames)
                obj.prms.(prmnames{l}) = obj.prms.(prmnames{l}) + obj.optm.adjust(obj.grad.(prmnames{l}), prmnames{l});
            end
            
            obj.SAFEDIV = obj.SAFEDIV*obj.DECAY;
        end
        
        function refresh(obj)
            obj.optm.refresh();
        end
        
        function PI = getPI(obj)
            q = softmax(obj.prms.p);
            PI = (q + obj.gam)./(sum(q) + obj.K*obj.gam);
        end
        
        function setoptm(obj, optm)
            obj.optm = optm;
            obj.optm.init(obj.prms);
        end
    end
end