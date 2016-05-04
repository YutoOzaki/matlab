function votingResults = votingSummary(patch,testResults,classNum)
    n_start = 1; n_end = n_start + patch - 1;
    testEntities = 0;
    voting = zeros(1,classNum);
    votingResults = 0;
    testSamples = size(testResults,1);
    
    confMat = zeros(classNum);
    confMatAll = zeros(classNum);
    
    while n_start<=testSamples
        testEntities = testEntities + 1;
        buf = testResults(n_start:n_end,1);

        for k=1:classNum
            voting(k) = length(find(k == buf));
        end
        [~,ind] = max(voting);

        l = testResults(n_start,2);
        if l == ind, votingResults = votingResults + 1; end

        confMat(l,ind) = confMat(l,ind) + 1;
        confMatAll(l,:) = confMatAll(l,:) + voting;
        
        n_start = n_end + 1;
        n_end = n_start + patch - 1;
    end
    
    fprintf('  voting results %d/%d (%3.2f%%)\n',votingResults,testEntities,100*votingResults/testEntities);
    disp(confMat);
    disp(confMatAll);
    
    figure(3);
    imshow(confMat,'initialMagnification',3600);title('CM (songs)');colormap('jet');caxis([0 classNum]), colorbar;
    figure(4);
    imshow(confMatAll,'initialMagnification',3600);title('CM (block-wise MFCC)');colormap('jet');caxis([0 patch*classNum]), colorbar;
    drawnow;
end