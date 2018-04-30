function bprop(nets, lossnode)
    dL = lossnode.backwardprop([]);
    L = size(dL.xmu,3);
    %lossnode.localgradcheck();
    
    priornet = nets.priornet;
    dgam = priornet.weight.backwardprop(dL);
    %deltacheck_gam(lossnode, priornet, L, dgam, 20);
    dzi = priornet.reparam.backwardprop(dgam);
    %deltacheck_zi(lossnode, priornet, L, dzi.sig, 20);
    
    decrpm = nets.decrpm;
    dxmu = decrpm.mu.backwardprop(L.*reshape(dL.xmu, [size(dL.xmu,1), size(dL.xmu,2)*L]));
    dxsig = decrpm.exp.backwardprop(L.*reshape(dL.xsig, [size(dL.xsig,1), size(dL.xsig,2)*L]));
    dlnxsig = decrpm.lnsigsq.backwardprop(dxsig);
    
    decnet = nets.decnet;
    names = flipud(fieldnames(decnet));
    delta = dxmu + dlnxsig;
    for i=1:length(names)
        delta = decnet.(names{i}).backwardprop(delta);
    end
    dxa = delta;
    
    %dxh = decnet.activate.backwardprop(dxmu + dlnxsig);
    %dxa = decnet.connect.backwardprop(dxh);
    
    encrpm = nets.encrpm;
    dzl = encrpm.reparam.backwardprop(reshape(dxa, [size(dxa,1), size(dxa,2)/L, L])./L);
    %deltacheck_zl(lossnode, encnet, decnet, L, dzl.sig, 20);
    
    dzmu = encrpm.mu.backwardprop(dL.zmu + dzi.mu + dzl.mu);
    dzsig = encrpm.exp.backwardprop(dL.zsig + dzi.sig + dzl.sig);
    dlnzsig = encrpm.lnsigsq.backwardprop(dzsig);
    
    encnet = nets.encnet;
    names = flipud(fieldnames(encnet));
    delta = dzmu + dlnzsig;
    for i=1:length(names)
        delta = encnet.(names{i}).backwardprop(delta);
    end
    
    %dzh = encnet.activate.backwardprop(dzmu + dlnzsig);
    %dza = encnet.connect.backwardprop(dzh);
end

function deltacheck_gam(lossnode, priornet, L, delta, num)
    fprintf('check dgam\n');

    input = lossnode.input;
    var = priornet.weight.input;
    
    eps = 1e-6;
    [J, N] = size(var);
    
    for k=1:num
        j = randi(J);
        n = randi(N);
        val = var(j, n);

        var(j, n) = val + eps;
        gam = priornet.weight.forwardprop(var);
        input.gam = gam;
        f1 = lossnode.forwardprop(input);

        var(j, n) = val - eps;
        gam = priornet.weight.forwardprop(var);
        input.gam = gam;
        f2 = lossnode.forwardprop(input);

        d1 = (f1 - f2)/(2*eps);
        d2 = delta(j, n)/N;
        re = abs(d1 - d2)/max(abs(d1), abs(d2));
        fprintf('debug:(%d, %d) %e, %e, %e\n', j, n, d1, d2, re);
        var(j, n) = val;
    end
    
    fprintf('\n');
end

function deltacheck_zi(lossnode, priornet, L, delta, num)
    fprintf('check dzi\n');

    input = lossnode.input;
    var = priornet.reparam.input;
    
    eps = 1e-6;
    [J, N] = size(var.sig);
    
    for k=1:num
        j = randi(J);
        n = randi(N);
        val = var.sig(j, n);

        var.sig(j, n) = val + eps;
        z = priornet.reparam.forwardprop(var);
        gam = priornet.weight.forwardprop(z);
        input.gam = gam;
        f1 = lossnode.forwardprop(input);

        var.sig(j, n) = val - eps;
        z = priornet.reparam.forwardprop(var);
        gam = priornet.weight.forwardprop(z);
        input.gam = gam;
        f2 = lossnode.forwardprop(input);

        d1 = (f1 - f2)/(2*eps);
        d2 = delta(j, n)/N;
        re = abs(d1 - d2)/max(abs(d1), abs(d2));
        fprintf('debug:(%d, %d) %e, %e, %e\n', j, n, d1, d2, re);
        var.sig(j, n) = val;
    end
    
    fprintf('\n');
end

function deltacheck_zl(lossnode, encnet, decnet, L, delta, num)
    fprintf('check dzl\n');
    
    input = lossnode.input;
    var = encnet.reparam.input;
    
    eps = 1e-6;
    [J, N] = size(var.sig);
    
    for k=1:num
        j = randi(J);
        n = randi(N);
        val = var.sig(j, n);

        var.sig(j, n) = val + eps;
        z = encnet.reparam.forwardprop(var);
        z = reshape(z, [size(z,1), size(z,2)*L]);
        xa = decnet.connect.forwardprop(z);
        xh = decnet.activate.forwardprop(xa);
        xmu = decnet.mu.forwardprop(xh);
        lnxsig = decnet.sig.forwardprop(xh);
        xsig = decnet.exp.forwardprop(lnxsig);
        xmu = reshape(xmu, [size(xmu,1), size(xmu,2)/L, L]);
        xsig = reshape(xsig, [size(xsig,1), size(xsig,2)/L, L]);

        input.xmu = xmu;
        input.xsig = xsig;
        f1 = lossnode.forwardprop(input);

        var.sig(j, n) = val - eps;
        z = encnet.reparam.forwardprop(var);
        z = reshape(z, [size(z,1), size(z,2)*L]);
        xa = decnet.connect.forwardprop(z);
        xh = decnet.activate.forwardprop(xa);
        xmu = decnet.mu.forwardprop(xh);
        lnxsig = decnet.sig.forwardprop(xh);
        xsig = decnet.exp.forwardprop(lnxsig);
        xmu = reshape(xmu, [size(xmu,1), size(xmu,2)/L, L]);
        xsig = reshape(xsig, [size(xsig,1), size(xsig,2)/L, L]);

        input.xmu = xmu;
        input.xsig = xsig;
        f2 = lossnode.forwardprop(input);

        d1 = (f1 - f2)/(2*eps);
        d2 = delta(j, n)/N;
        re = abs(d1 - d2)/max(abs(d1), abs(d2));
        fprintf('debug:(%d, %d) %e, %e, %e\n', j, n, d1, d2, re);
        var.sig(j, n) = val;
    end
    
    fprintf('\n');
end