classdef lossfunc < basenode
    properties
        input, prms, delta
    end
    
    methods
        function obj = lossfunc()
            obj.prms = [];
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
            eta= input.eta;
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
                A = bsxfun(@minus, zmu, eta(k).mu).^2;
                B = bsxfun(@rdivide, A, eta(k).sig);
                C = bsxfun(@rdivide, zsig, eta(k).sig) + B;
                D = bsxfun(@plus, log(2*pi.*eta(k).sig), C);

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
            K = length(obj.input.PI);
            [~, batchsize] = size(obj.input.x);
            eta = obj.input.eta;
            gam = obj.input.gam;
            zmu = obj.input.zmu;
            zsig = obj.input.zsig;
            PI = obj.input.PI;
            
            dgam = zeros(K, batchsize);
            for k=1:K
                buf = log(bsxfun(@rdivide, PI(k), gam(k,:)));
                idx = buf == Inf;
                buf(idx) = log(PI(k)/1e-5);
                
                A = bsxfun(@minus, zmu, eta(k).mu).^2;
                B = bsxfun(@rdivide, A, eta(k).sig);
                C = bsxfun(@rdivide, zsig, eta(k).sig) + B;
                D = bsxfun(@plus, log(2*pi.*eta(k).sig), C);
                
                dgam(k, :) = -0.5.*sum(D) + buf - 1;
            end
            
            dPI = bsxfun(@rdivide, gam, PI);
            
            delta = struct(...
                'gam', dgam,...
                'PI', dPI...
                );
            
            obj.delta = delta;
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
            names = {'gam', 'PI'};
            
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
                
                if strcmp(names{i}, 'gam')
                    d(2) = obj.delta.(names{i})(ix, iy)./batchsize;
                elseif strcmp(names{i}, 'PI')
                    d(2) = sum(obj.delta.(names{i})(ix, :))./batchsize;
                end
                
                re = abs(d(1) - d(2))./max(abs(d(1)), abs(d(2)));
                fprintf('%s: %e, %e, %e, %e\n', names{i}, val, d(1), d(2), re);
            end
            
            obj.input = cache;
        end
    end
end