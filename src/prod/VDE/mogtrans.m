classdef mogtrans < basenode
    properties
        input, prms, grad, optm
        K, J, gam
    end
    
    methods
        function obj = mogtrans(K, J, gam, optm)
            obj.K = K;
            obj.J = J;
            obj.gam = gam;
            
            obj.prms = struct(...
                'eta_mu', zeros(obj.J, obj.K),...
                'eta_lnsig', zeros(obj.J, obj.K),...
                'p', zeros(obj.K, 1)...
                );
            
            optm.ms = obj.prms;
            obj.optm = optm;
        end
        
        function [output, r] = forwardprop(obj, input)
            PI = obj.getPI();
            eta_sig = exp(obj.prms.eta_lnsig);
            
            obj.input = input;
            [~, batchsize] = size(input);
            
            r = zeros(obj.K, batchsize);
            for k=1:obj.K
                r(k, :) = PI(k) .* mvnpdf(input', obj.prms.eta_mu(:, k)', diag(eta_sig(:, k)));
            end
            
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
            
            z_kpdf = zeros(obj.K, batchsize);
            for k=1:obj.K
                z_kpdf(k, :) = mvnpdf(obj.input', eta_mu(:, k)', diag(eta_sig(:, k)))';
            end
            
            dgamdPI = zeros(obj.K, obj.K, batchsize);
            dgamdeta_mu = zeros(obj.K, obj.K, obj.J, batchsize);
            dgamdeta_sig = zeros(obj.K, obj.K, obj.J, batchsize);
            dgamdz = zeros(obj.K, obj.J, batchsize);
            I = sum(bsxfun(@times, PI, z_kpdf));
            M = bsxfun(@minus, reshape(obj.input, [obj.J,1,batchsize]), eta_mu);
            N = bsxfun(@rdivide, M, eta_sig);
            O = bsxfun(@times, reshape(-PI, [1,obj.K,1]), N);
            P = bsxfun(@times, reshape(z_kpdf, [1,obj.K,batchsize]), O);
            Q = squeeze(sum(P, 2));
            normalization = I.^2;
            
            for k=1:obj.K
                idx = setdiff(1:obj.K, k);
                buf = sum(bsxfun(@times, PI(idx), z_kpdf(idx, :)));
                
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
                    dgamdPI(k, idx(i), :) = -PI(k).*z_kpdf(k, :).*z_kpdf(idx(i), :)./normalization;
                    
                    A = -PI(idx(i)).*z_kpdf(idx(i), :).*PI(k).*z_kpdf(k, :)./normalization;
                    
                    B = bsxfun(@minus, obj.input, eta_mu(:, idx(i)));
                    C = bsxfun(@rdivide, B, eta_sig(:, idx(i)));
                    dgamdeta_mu(k, idx(i), :, :) = bsxfun(@times, A, C);
                    
                    C = bsxfun(@plus, -eta_sig(:, idx(i)), B.^2);
                    D = bsxfun(@rdivide, C, 2.*eta_sig(:, idx(i)).^2);
                    dgamdeta_sig(k, idx(i), :, :) = bsxfun(@times, A, D);
                end
            end
            
            gPI = zeros(obj.K, 1) + dLdPI;
            geta_mu = zeros(obj.J, obj.K) + dLdeta_mu;
            geta_sig = zeros(obj.J, obj.K) + dLdeta_sig;
            delta = zeros(obj.J, batchsize);
            for n=1:batchsize
                gPI = gPI + (dLdgam(:, n)' * dgamdPI(:, :, n))';
                delta(:, n) = (dLdgam(:, n)' * dgamdz(:, :, n))';
                
                for k=1:obj.K
                    buf = squeeze(dgamdeta_mu(:, k, :, n));
                    geta_mu(:, k) = geta_mu(:, k) + (dLdgam(:, n)' * buf)';
                    
                    buf = squeeze(dgamdeta_sig(:, k, :, n));
                    geta_sig(:, k) = geta_sig(:, k) + (dLdgam(:, n)' * buf)';
                end
            end
            
            geta_lnsig = geta_sig.*exp(obj.prms.eta_lnsig);
            
            dPIdq = zeros(obj.K, obj.K);
            dqdp = zeros(obj.K, obj.K);
            for i=1:obj.K
                j = i;
                idx = setdiff(1:obj.K, i);
                
                dPIdq(i, j) = sum(q(idx))+ (obj.K - 1).*obj.gam;
                dqdp(i, j) =  exp(obj.prms.p(i))*sum(exp(obj.prms.p(idx)));
                
                for j=1:(obj.K-1)
                    dPIdq(i, idx(j)) = -q(i) - obj.gam;
                    dqdp(i, idx(j)) = -exp(obj.prms.p(i) + obj.prms.p(idx(j)));
                end
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
            obj.prms.p = rand(obj.K, 1);
            
            for k=1:obj.K
                obj.prms.eta_mu(:, k) = rand(obj.J, 1)';
                obj.prms.eta_lnsig(:, k) = rand(obj.J, 1)';
                %obj.prms.eta_mu(:, k) = mvnrnd(zeros(1, obj.J), diag(3.*ones(obj.J, 1)), 1)';
                %obj.prms.eta_lnsig(:, k) = mvnrnd(zeros(1, obj.J), diag(ones(obj.J, 1)), 1)';
            end
        end
        
        function update(obj)
            prmnames = fieldnames(obj.grad);
            
            for l=1:length(prmnames)
                obj.prms.(prmnames{l}) = obj.prms.(prmnames{l}) + obj.optm.adjust(obj.grad.(prmnames{l}), prmnames{l});
            end
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
            obj.optm.ms = obj.prms;
            obj.optm.refresh();
        end
    end
end