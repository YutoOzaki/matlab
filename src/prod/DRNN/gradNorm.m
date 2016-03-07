function nrm = gradNorm(gprms)
    L = length(gprms);
    nrm = zeros(1,L);
    
    for i=1:L
        nrm(i) = norm(gprms{i},2);
    end
end