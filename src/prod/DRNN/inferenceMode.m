function results = inferenceMode(dataSet, nnet, classNum, longTerm, i)
    tic;
    result_max = 0;
    result_mean = 0;
    result_maxRate = 0;
    loss_training = 0;
    L = length(nnet);
    T = nnet{1}.T;
    h = zeros(classNum,dataSet.batchSize);
    confmat = zeros(classNum,classNum,3);
    
    for l=1:L
        nnet{l}.dropoutCompensation();
    end
    
    while dataSet.setNextIndex();
        batchData = dataSet.getFeature();
        label = dataSet.getLabel()';
        labelVector = dataSet.label2vec(label, classNum, T);

        T_total = size(batchData, 3);
        t = 1;
        mean_buf = 0;
        
        if longTerm
            for l=1:L
                nnet{l}.resetStates();
            end
        end
        
        while t+T-1 <= T_total
            input = batchData(:,:,t:t+T-1);
            for l=1:L
                if longTerm
                    nnet{l}.continueStates();
                end
                input = nnet{l}.fprop(input);
            end
            
            for k=1:T
                [~, mind] = max(input(:,:,k));
                for j=1:length(mind)
                    h(mind(j),j) = h(mind(j),j) + 1;
                end
            end
            
            mean_buf = mean_buf + input;
            loss_training = loss_training + labelVector.*log(input);
            t = t + T;
        end
        
        [~, mind] = max(mean(mean_buf,3));
        result_mean = result_mean + length(find((mind - label) == 0));
        
        for j=1:dataSet.batchSize
            confmat(label(j),mind(j),1) = confmat(label(j),mind(j),1) + 1;
        end
        
        [~, mind] = max(input(:,:,end));
        result_max = result_max + length(find((mind - label) == 0));
        
        for j=1:dataSet.batchSize
            confmat(label(j),mind(j),2) = confmat(label(j),mind(j),2) + 1;
        end
        
        [~, mind] = max(h);
        result_maxRate = result_maxRate + length(find((mind - label) == 0));
        
        for j=1:dataSet.batchSize
            confmat(label(j),mind(j),3) = confmat(label(j),mind(j),3) + 1;
        end
        
        %{
        h = h./(t-1);
        r = 1 + sum(h.*log(h+eps))./log(classNum);
        %}
        h = h.*0;
    end
    loss_training = sum(sum(sum(loss_training)));
    t = toc;
    
    for l=1:L
        nnet{l}.dropoutCompensation();
    end

    results = 100.*[result_mean; result_max; result_maxRate]./dataSet.N;
    
    fprintf('elapsed time %3.3f (inference on %s data set)\n', t, dataSet.dataset);
    fprintf('cross-entropy loss %5.3f\n', loss_training);
    fprintf('classification result (mean)      %d/%d (%3.3f%%)\n', result_mean, dataSet.N, results(1));
    fprintf('classification result (max)       %d/%d (%3.3f%%)\n', result_max, dataSet.N, results(2));
    fprintf('classification result (max rate)  %d/%d (%3.3f%%)\n', result_maxRate, dataSet.N, results(3));
    
    if dataSet.N > 0
        subplot(3,3,(i-1)*3+1);
        imshow(confmat(:,:,1));title('mean');colormap('jet');caxis([0 dataSet.N/classNum]);
        subplot(3,3,(i-1)*3+2);
        imshow(confmat(:,:,2));title('max');colormap('jet');caxis([0 dataSet.N/classNum]);
        subplot(3,3,(i-1)*3+3);
        imshow(confmat(:,:,3));title('max rate');colormap('jet');caxis([0 dataSet.N/classNum]);
        drawnow
    end
end