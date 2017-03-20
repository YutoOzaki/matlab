function tests = deleteElementTest
    tests = functiontests(localfunctions);
end

function testDeletingHead(testCase)
    array = [3 4 5 6 7 8 9 10 0.5];
    array = array(:);
    k = 1;
    
    actSolution = utils.deleteElement(array, k);
    
    expSolution = [4 5 6 7 8 9 10 0.5];
    expSolution = expSolution(:);
    
    verifyEqual(testCase, actSolution, expSolution);
end

function testDeletingTail(testCase)
    array = [3 4 5 6 7 8 9 10 0.5];
    array = array(:);
    k = 9;
    
    actSolution = utils.deleteElement(array, k);
    
    expSolution = [3 4 5 6 7 8 9 10];
    expSolution = expSolution(:);
    
    verifyEqual(testCase, actSolution, expSolution);
end

function testDeletingMiddle(testCase)
    array = [3 4 5 6 7 8 9 10 0.5];
    array = array(:);
    k = 4;
    
    actSolution = utils.deleteElement(array, k);
    
    expSolution = [3 4 5 7 8 9 10 0.5];
    expSolution = expSolution(:);
    
    verifyEqual(testCase, actSolution, expSolution);
end

function testDeletingMiddle2D(testCase)
    array = [3 4 5 6 7 8 9 10 0.5;
             1 2 3 4 5 6 7 80 0.1;
             9 8 7 6 5 4 3 20 0.6];
         
    k = 2;
    
    actSolution = utils.deleteElement(array, k);
    
    expSolution = [3 4 5 6 7 8 9 10 0.5;
                   9 8 7 6 5 4 3 20 0.6];
    
    verifyEqual(testCase, actSolution, expSolution);
end

function testCRPCase(testCase)
    K = 8;
    array = [3 4 5 6 7 8 9 10 0.5];
    array = array(:);
    assert((K + 1) == length(array), 'test data is inconsistent with K');
    
    k = 4;
    
    actSolution = utils.deleteElement(array, k);
    K = K - 1;
    
    expSolution = [3 4 5 7 8 9 10 0.5];
    expSolution = expSolution(:);
    
    assert((K + 1) == length(actSolution), 'length of n_k is inconsistent with K');
    verifyEqual(testCase, actSolution, expSolution);
end

function testSBPTreeCase(testCase)
    N = 50;
    eta = [2 1];
    z = zeros(N, 1);
    z = ncrp(eta, z);

    alpha = [4 3 2 1];
    K = 10;
    pi = sbptree(alpha, K, z);
    
    k = 4;
    actSolution = utils.deleteElement(pi, k);
    K = K - 1;
    
    L = length(alpha);
    for l=1:L
        verifyEqual(testCase, size(actSolution{l}, 1), K);
        
        verifyEqual(testCase, actSolution{l}(k - 1, :), pi{l}(k - 1, :));
        verifyEqual(testCase, actSolution{l}(k, :)    , pi{l}(k + 1, :));
        verifyEqual(testCase, actSolution{l}(k + 1, :), pi{l}(k + 2, :));
    end
end

function setupOnce(testCase)
end

function teardown(testCase)
end