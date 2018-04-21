function gradcheck(x, nets, lossnode)
    eps = 1e-6;
    f = zeros(2, 1);
    d = zeros(2, 1);
 
    netnames = fieldnames(nets);
    L = length(netnames);
    
    for l=1:L
        names = fieldnames(nets.(netnames{l}));

        for i=1:length(names)
            prms = nets.(netnames{l}).(names{i}).prms;
            
            if isempty(prms)
                J = 0;
            else
                prmnames = fieldnames(prms);
                J = length(prmnames);
            end

            for j=1:J
                prm = prms.(prmnames{j});

                m = size(prm, 1);
                n = size(prm, 2);
                a = randi(m);
                b = randi(n);
                val = prm(a, b);

                prm(a, b) = val + eps;
                nets.(netnames{l}).(names{i}).prms.(prmnames{j}) = prm;
                f(1) = fprop(x, nets, lossnode);

                prm(a, b) = val - eps;
                nets.(netnames{l}).(names{i}).prms.(prmnames{j}) = prm;
                f(2) = fprop(x, nets, lossnode);

                d(1) = (f(1) - f(2))/(2*eps);
                d(2) = nets.(netnames{l}).(names{i}).grad.(prmnames{j})(a, b);

                re = abs(d(1) - d(2))/max(abs(d(1)), abs(d(2)));
                fprintf('%s.%s.%s, %e, %e, %e, %e\n', netnames{l}, names{i}, prmnames{j}, val, d(1), d(2), re);

                prm(a, b) = val;
                nets.(netnames{l}).(names{i}).prms.(prmnames{j}) = prm;
            end
        end
    end
    
    fprintf('\n');
end