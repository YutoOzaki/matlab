function gradcheck(x, encnet, decnet, priornet, lossnode)
    eps = 1e-6;
    f = zeros(2, 1);
    d = zeros(2, 1);

    names = fieldnames(priornet);
    names = {'weight'};
    
    for i=1:length(names)
        prms = priornet.(names{i}).prms;
        prmnames = fieldnames(prms);
        prmnames = {'PI'};
        
        for j=1:length(prmnames)
            prm = prms.(prmnames{j});
            
            m = size(prm, 1);
            n = size(prm, 2);
            a = randi(m);
            b = randi(n);
            val = prm(a, b);
            
            prm(a, b) = val + eps;
            priornet.(names{i}).prms.(prmnames{j}) = prm;
            f(1) = fprop(x, encnet, decnet, priornet, lossnode);

            prm(a, b) = val - eps;
            priornet.(names{i}).prms.(prmnames{j}) = prm;
            f(2) = fprop(x, encnet, decnet, priornet, lossnode);

            d(1) = (f(1) - f(2))/(2*eps);
            d(2) = priornet.(names{j}).delta.(prmnames{j})(a, b);
            
            re = abs(d(1) - d(2))/max(abs(d(1)), abs(d(2)));
            fprintf('%s, %e, %e, %e, %e\n', prmnames{j}, val, d(1), d(2), re);
            
            prm(a, b) = val;
            priornet.(names{i}).prms.(prmnames{j}) = prm;
        end
    end
end