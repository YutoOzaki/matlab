function RunDRNN(name)
    %% check machine
    if ispc==1
        fprintf('--Running on Windows machine ');
    elseif isunix==1
        fprintf('--Running on Linux or Mac OS machine ');
    else
        warning('--This machine is neither Windows, Linux, nor Mac OS ');
    end

    gpuInfo = gpuDevice;
    if gpuInfo.DeviceSupported && strcmp(gpuInfo.ComputeMode,'Default')
        fprintf('(GPU is available)--\n');
    else
        fprintf('(GPU is not available)--\n');
    end

    %% get hyperparameters
    hprms = modeler();
    
    %% logging
    homeDir = userpath;
    homeDir = homeDir(1:end - 1);
    dev = '\dev\GPU_test\DRNN';
    log = '\logs\';
    logDir = strrep(strcat(homeDir,dev,log), '\', '/');

    %% run deep recurrent neural network
    for i=1:length(hprms)
        diary off
        timeStamp = clock;
        timeStr = '';
        for j=1:length(timeStamp)-1
            if j~= 1
                timeBuf = num2str(timeStamp(j),'%02i');
            else
                timeBuf = num2str(timeStamp(j));
            end
            timeStr = strcat(timeStr,timeBuf);
        end
        diary(strcat(logDir, name, '_', timeStr, '.txt'));
        
        diary on
        result = DRNN(hprms(i));
        diary off
        
        fid = fopen(strcat(logDir, 'reports-', name, '.txt'), 'a');
        fprintf(fid, '%s\n', strcat(name, '_', timeStr));
        for j=1:size(result,1)
            fprintf(fid, '%3.3f%% %d\n', result(j,1), result(j,2));
        end
        fclose(fid);
    end
end