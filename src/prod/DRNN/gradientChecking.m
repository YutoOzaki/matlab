function relativeError = gradientChecking(l,gprms,d,meps,dataMat,fprops,expand,weiComp,prms,auxPrms,T)
    bufPrms = prms;
    batchSize = size(dataMat,2);
    dMat = repmat(d,[1 1 T]);
    meps = sqrt(meps);
    prmNum = length(prms{l});
    relativeError = zeros(prmNum,3);
    
    for i=1:prmNum
        prm = prms{l}{i};
        eps = meps*abs(prm(1,1));
        
        prm(1,1) = prm(1,1) + eps;
        bufPrms{l}{i} = prm;
        output = fpropBatch(dataMat,batchSize,fprops,expand,weiComp,bufPrms,auxPrms,T);
        dy1 = sum(sum(log(output).*dMat,3),1);
        
        prm = prms{l}{i};
        prm(1,1) = prm(1,1) - eps;
        bufPrms{l}{i} = prm;
        output = fpropBatch(dataMat,batchSize,fprops,expand,weiComp,bufPrms,auxPrms,T);
        dy2 = sum(sum(log(output).*dMat,3),1);
        
        dt1 = (-dy1 + dy2)/(2*eps);
        dt1 = mean(dt1);
        dt2 = gprms{i}(1,1);
        
        relativeError(i,1) = dt1;
        relativeError(i,2) = dt2;
        relativeError(i,3) = abs(dt1 - dt2)/max(abs(dt1),abs(dt2));
    end
end