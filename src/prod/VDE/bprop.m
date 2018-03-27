function bprop(encnet, decnet, priornet, lossnode)
    dL = lossnode.backwardprop([]);
    %lossnode.localgradcheck();
    
    dgam = priornet.weight.backwardprop(dL);
    dzi = priornet.reparam.backwardprop(dgam);
    
    L = size(dL.xmu,3);
    dxmu = decnet.mu.backwardprop(L.*reshape(dL.xmu, [size(dL.xmu,1), size(dL.xmu,2)*L]));
    dxsig = decnet.exp.backwardprop(L.*reshape(dL.xsig, [size(dL.xsig,1), size(dL.xsig,2)*L]));
    dlnxsig = decnet.sig.backwardprop(dxsig);
    dxh = decnet.activate.backwardprop(dxmu + dlnxsig);
    dxa = decnet.connect.backwardprop(dxh);
    
    dzl = encnet.reparam.backwardprop(reshape(dxa, [size(dxa,1), size(dxa,2)/L, L])./L);
    
    dzsig = encnet.exp.backwardprop(dzi.sig + dzl.sig);
    dlnzsig = encnet.sig.backwardprop(dzsig);
    dzmu = encnet.mu.backwardprop(dzi.mu + dzl.mu);
    dzh = encnet.activate.backwardprop(dzmu + dlnzsig);
    dza = encnet.connect.backwardprop(dzh);
end