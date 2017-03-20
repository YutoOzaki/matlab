function tests = crpTest
    tests = functiontests(localfunctions);
end

function testInit(testCase)
    alpha = 1;
    
    z = zeros(1000, 1);
    
    [z, n_k, K] = crp(alpha, z);
    K_mean = utils.expcrp(alpha, length(z));
    fprintf('CRP: Number of clusters %d (%3.3f as mean)\n', length(n_k), K_mean);
    
    commonVerification(testCase, z, n_k, K)
    commonPlot(n_k, K)
end

function testResampleWithAllSame(testCase)
    alpha = 1;
    
    z = ones(1000, 1);
    
    [z, n_k, K] = crp(alpha, z);
    
    commonVerification(testCase, z, n_k, K)
    commonPlot(n_k, K)
end

function testResampleWithAllDiff(testCase)
    alpha = 1;
    
    z = 1:1000;
    
    [z, n_k, K] = crp(alpha, z);
    
    commonVerification(testCase, z, n_k, K)
    commonPlot(n_k, K)
end

function testDenseDist(testCase)
    alpha = 20;
    
    z = zeros(10000, 1);
    
    [z, n_k, K] = crp(alpha, z);
    K_mean = utils.expcrp(alpha, length(z));
    fprintf('CRP: Number of clusters %d (%3.3f as mean)\n', length(n_k), K_mean);
    
    commonVerification(testCase, z, n_k, K)
    commonPlot(n_k, K)
end

function testSparseDist(testCase)
    alpha = 0.01;
    
    z = zeros(10000, 1);
    
    [z, n_k, K] = crp(alpha, z);
    K_mean = utils.expcrp(alpha, length(z));
    fprintf('CRP: Number of clusters %d (%3.3f as mean)\n', length(n_k), K_mean);
    
    commonVerification(testCase, z, n_k, K)
    commonPlot(n_k, K)
end

function setupOnce(testCase)
end

function teardown(testCase)
end

function commonVerification(testCase, z, n_k, K)
    k = unique(z);
    utils.checkSequence(k, 'k');
    
    verifyEqual(testCase, sum(n_k), length(z));
    verifyEqual(testCase, length(k), K);
    
    verifyGreaterThanOrEqual(testCase, z, 1);
    verifyGreaterThanOrEqual(testCase, n_k, 1);
end

function commonPlot(n_k, K)
    figure;
    
    stem(n_k, 'Marker', 'None');
    title(sprintf('number of customers assigned to each table (K = %d)', K));
    
    fig = gcf;
    ax = fig.CurrentAxes;
    ax.XLim = [-1 length(n_k)+1];
    ax.XTick = [];
end