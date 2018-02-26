classdef mogtrans < basenode
    properties
        input, prms, delta
    end
    
    methods
        function obj = mogtrans(eta, PI)
            obj.prms = struct('eta', eta, 'PI', PI);
        end
        
        function output = forwardprop(obj, input)
            obj.input = input;
            K = length(obj.prms.PI);
            [~, batchsize] = size(input);
            
            output = zeros(K, batchsize);
            for k=1:K
                output(k, :) = obj.prms.PI(k) .* mvnpdf(input', obj.prms.eta(k).mu', diag(obj.prms.eta(k).sig));
            end
            output = bsxfun(@rdivide, output, sum(output));
            
            idx = abs(output) < 1e-5;
            output(idx) = 1e-5;
        end
        
        function delta = backwardprop(obj, input)
            K = length(obj.prms.PI);
            [~, batchsize] = size(obj.input);
            
            dLdgam = input.gam;
            dLdPI = input.PI;
            eta = obj.prms.eta;
            PI = obj.prms.PI;
            
            z_kpdf = zeros(K, batchsize);
            for k=1:K
                z_kpdf(k, :) = mvnpdf(obj.input', eta(k).mu', diag(eta(k).sig))';
            end
            
            dgamdPI = zeros(K, K, batchsize);
            normalization = sum(bsxfun(@times, PI, z_kpdf)).^2;
            for i=1:K
                idx = setdiff(1:K, i);
                buf = sum(bsxfun(@times, PI(idx), z_kpdf(idx, :)));
                dgamdPI(i, i, :) = z_kpdf(i, :).*buf./normalization;
                
                for k=1:(K-1)
                    buf = -PI(idx(k)).*z_kpdf(idx(k), :);
                    dgamdPI(idx(k), i, :) = buf.*z_kpdf(i, :)./normalization;
                end
            end
            
            gPI = zeros(K, 1);
            for n=1:batchsize
                gPI = gPI + dLdPI(:, n) + (dLdgam(:, n)' * dgamdPI(:, :, n))';
            end
            gPI = gPI./batchsize;
            
            delta = struct(...
                'PI', gPI...
                );
            
            obj.delta = delta;
        end
        
        function init(obj)
        end
        
        function update(obj)
        end
    end
end