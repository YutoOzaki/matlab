classdef mogtrans < basenode
    properties
        input, prms, grad, optm
    end
    
    methods
        function obj = mogtrans(eta_mu, eta_r, PI, optm)
            obj.prms = struct('eta_mu', eta_mu, 'eta_r', eta_r, 'PI', PI);
            
            obj.optm = optm;
        end
        
        function output = forwardprop(obj, input)
            obj.input = input;
            K = length(obj.prms.PI);
            [~, batchsize] = size(input);
            
            eta_sig = exp(obj.prms.eta_r);
            
            output = zeros(K, batchsize);
            for k=1:K
                output(k, :) = obj.prms.PI(k) .* mvnpdf(input', obj.prms.eta_mu(:, k)', diag(eta_sig(:, k)));
            end
            output = bsxfun(@rdivide, output, sum(output));
        end
        
        function delta = backwardprop(obj, input)
            K = length(obj.prms.PI);
            [J, batchsize] = size(obj.input);
            
            dLdgam = input.gam;
            dLdPI = input.PI;
            dLdeta_mu = input.eta_mu;
            dLdeta_sig = input.eta_sig;
            
            eta_mu = obj.prms.eta_mu;
            eta_sig = exp(obj.prms.eta_r);
            PI = obj.prms.PI;
            
            z_kpdf = zeros(K, batchsize);
            for k=1:K
                z_kpdf(k, :) = mvnpdf(obj.input', eta_mu(:, k)', diag(eta_sig(:, k)))';
            end
            
            dgamdPI = zeros(K, K, batchsize);
            dgamdeta_mu = zeros(K, K, J, batchsize);
            dgamdeta_sig = zeros(K, K, J, batchsize);
            dgamdz = zeros(K, J, batchsize);
            I = sum(bsxfun(@times, PI, z_kpdf));
            M = bsxfun(@minus, reshape(obj.input, [J,1,batchsize]), eta_mu);
            N = bsxfun(@rdivide, M, eta_sig);
            O = bsxfun(@times, reshape(-PI, [1,K,1]), N);
            P = bsxfun(@times, reshape(z_kpdf, [1,K,batchsize]), O);
            Q = squeeze(sum(P, 2));
            normalization = I.^2;
            
            for i=1:K
                idx = setdiff(1:K, i);
                buf = sum(bsxfun(@times, PI(idx), z_kpdf(idx, :)));
                
                dgamdPI(i, i, :) = z_kpdf(i, :).*buf./normalization;
                
                A = bsxfun(@minus, obj.input, eta_mu(:, i));
                B = bsxfun(@rdivide, A, eta_sig(:, i));
                C = bsxfun(@times, B, z_kpdf(i, :));
                D = bsxfun(@rdivide, C, normalization);
                
                dgamdeta_mu(i, i, :, :) = bsxfun(@times, PI(i).*buf, D);
                
                E = bsxfun(@plus, -eta_sig(:, i), A.^2);
                F = bsxfun(@rdivide, E, 2.*eta_sig(:, i).^2);
                G = bsxfun(@times, F, z_kpdf(i, :));
                H = bsxfun(@rdivide, G, normalization);
                
                dgamdeta_sig(i, i, :, :) = bsxfun(@times, PI(i).*buf, H);
                
                R = bsxfun(@times, -PI(i).*C, I) - bsxfun(@times, PI(i).*z_kpdf(i, :), Q);
                dgamdz(i, :, :) = bsxfun(@rdivide, R, normalization);
                
                for k=1:(K-1)
                    buf = -PI(idx(k)).*z_kpdf(idx(k), :);
                    
                    dgamdPI(idx(k), i, :) = buf.*z_kpdf(i, :)./normalization;
                    dgamdeta_mu(idx(k), i, :, :) = PI(i).*bsxfun(@times, D, buf);
                    dgamdeta_sig(idx(k), i, :, :) = PI(i).*bsxfun(@times, H, buf);
                end
            end
            
            gPI = zeros(K, 1);
            geta_mu = zeros(J, K);
            geta_sig = zeros(J, K);
            delta = zeros(J, batchsize);
            for n=1:batchsize
                gPI = gPI + dLdPI(:, n) + (dLdgam(:, n)' * dgamdPI(:, :, n))';
                
                geta_mu = geta_mu + dLdeta_mu(:, :, n);
                geta_sig = geta_sig + dLdeta_sig(:, :, n);
                
                delta(:, n) = (dLdgam(:, n)' * dgamdz(:, :, n))';
                
                for k=1:K
                    buf = squeeze(dgamdeta_mu(:, k, :, n));
                    geta_mu(:, k) = geta_mu(:, k) + (dLdgam(:, n)' * buf)';
                    
                    buf = squeeze(dgamdeta_sig(:, k, :, n));
                    geta_sig(:, k) = geta_sig(:, k) + (dLdgam(:, n)' * buf)';
                end
            end
            
            geta_r = geta_sig.*exp(obj.prms.eta_r);
            
            gPI = gPI./batchsize;
            geta_mu = geta_mu./batchsize;
            geta_r = geta_r./batchsize;
            
            obj.grad = struct(...
                'PI', gPI,...
                'eta_mu', geta_mu,...
                'eta_r', geta_r...
                );
        end
        
        function init(obj)
        end
        
        function update(obj)
            prmnames = fieldnames(obj.grad);
            
            for l=1:length(prmnames)
                obj.prms.(prmnames{l}) = obj.prms.(prmnames{l}) + obj.optm.adjust(obj.grad.(prmnames{l}));
            end
        end
    end
end