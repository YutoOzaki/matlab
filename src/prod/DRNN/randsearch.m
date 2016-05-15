function hprms = randsearch()
    home = userpath;
    home = home(1:end-1);
    trials = 120;
    classNum = 10;
    
    hprms = struct(...
        'nhid',           {},....
        'units',          {},...
        'optimization',   {},...
        'batchSize',      {},...
        'baselr',         {},...
        'epochs',         {},...
        'drate',          {},...
        'droprate',       {},...
        'gcheck',         {},...        
        'dataPrep',       {},...     
        'patch',          {}...
    );

    hprms(trials).gcheck = false;
    
    a = log(1e-5);
    b = log(5*1e-1);
    baselr = exp((b-a).*rand(trials,1) + a);
    
    a = 0.8;
    b = 0.99;
    drate = (b-a).*rand(trials,1) + a;
    
    a = 0.4;
    b = 0.6;
    droprate = [ones(trials,1) 1 - binornd(1,0.5,trials,1).*((b-a).*rand(trials,1) + a)];
    
    a = log(10);
    b = log(20);
    epochs = round(exp((b-a).*rand(trials,1) + a));
    
    a = [10 20 40 80 100 125 200];
    b = randi(length(a),trials,1);
    batchSize = a(b)';
    
    unitType = {'mcell','bmcell'};
    nhid = cell(1,trials);
    units = cell(1,trials);
    a = log(10);
    b = log(256);
    unitnum = round(exp((b-a).*rand(trials,1) + a));
    for i=1:trials
        buf = unitType{randi(2)};
        units{i} = {buf,'smax'};
        
        if strcmp(buf,'mcell')
            nhid{i} = {unitnum(i) classNum};
        elseif strcmp(buf,'bmcell')
            nhid{i} = {[unitnum(i) unitnum(i)] classNum};
        end
    end
    
    for i=1:trials
        hprms(i).nhid          = nhid{i};
        hprms(i).units         = units{i};
        hprms(i).optimization  = 'rmsprop';
        hprms(i).batchSize     = batchSize(i);
        hprms(i).baselr        = baselr(i);
        hprms(i).epochs        = epochs(i);
        hprms(i).drate         = drate(i);
        hprms(i).droprate      = droprate(i,:);
        hprms(i).gcheck        = false;        
        hprms(i).dataPrep      = @(x) prepMFCC(strcat(home,'/data/gtzan/gtzanMFCC.mat'));
        hprms(i).patch         = 120;
    end
end