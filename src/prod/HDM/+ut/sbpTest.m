function tests = sbpTest
    tests = functiontests(localfunctions);
end

function testSmallRand(testCase)
    alpha = 1;
    
    K = 10;
    
    actSolution = sbp(alpha, K);
    
    commonVerification(actSolution, testCase);
    
    commonPlot(actSolution);
end

function testLargeRandQuality(testCase)
    alpha = 1;
    
    K = 10000;
    
    for i=1:1000
        x = rand(10000, 1);
        x = x./sum(x);

        actSolution = sbp(alpha, K);

        commonVerification(actSolution, testCase);
    end

    commonPlot(actSolution);
end

function testDenseDist(testCase)
    alpha = 20;
    
    K = 100;
    
    actSolution = sbp(alpha,K);
    
    commonVerification(actSolution, testCase);
    
    commonPlot(actSolution);
end

function testSparseDist(testCase)
    alpha = 0.01;
    
    K = 100;
    
    actSolution = sbp(alpha,K);
    
    commonVerification(actSolution, testCase);
    
    commonPlot(actSolution);
end

function setupOnce(testCase)
end

function teardown(testCase)
end

function commonVerification(actSolution, testCase)
    verifyGreaterThanOrEqual(testCase, actSolution, 0, ...
        @() disp(actSolution(end)));
    
    import matlab.unittest.constraints.HasNaN
    verifyNotEqual(testCase, actSolution, HasNaN, ...
        'The probability vector contains NaN');
    
    verifyEqual(testCase, imag(actSolution), zeros(length(actSolution), 1), 'AbsTol', eps,...
        'The probability vector contains imaginary numbers');
    
    verifyEqual(testCase, sum(actSolution), 1);
end

function commonPlot(actSolution)
    figure;
    
    subplot(211);
    stem(actSolution, 'Marker', 'None');
    title('Stick breaking weights');
    
    fig = gcf;
    ax = fig.CurrentAxes;
    ax.XLim = [-1 length(actSolution)+1];
    ax.XTick = [];
    
    subplot(212);
    [y, h] = getNormRndBase(actSolution);
    stem(h, y, 'Marker', 'None');
    title('Realization of the base measure');
    
    fig = gcf;
    ax = fig.CurrentAxes;
    ax.XTick = [];
end

function [y, h] = getNormRndBase(x)
    ind = mnrnd(1, repmat(x', 500, 1));
    [~, col] = find(ind);
    
    K = length(x);
    h = normrnd(0, 1, [K 1]);
    
    pd = makedist('Normal', 0, 1);
    y = pdf(pd, h(col));
    
    h = h(col);
end