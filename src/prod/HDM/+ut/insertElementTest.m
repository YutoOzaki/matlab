function tests = insertElementTest
    tests = functiontests(localfunctions);
end

function testInsertingToHead(testCase)
    array = [3 4 5 6 7 8 9 10 0.5];
    array = array(:);
    element = 1;
    k = 1;
    
    actSolution = utils.insertElement(array, element, k);
    
    expSolution = [1 3 4 5 6 7 8 9 10 0.5];
    expSolution = expSolution(:);
    
    verifyEqual(testCase, actSolution, expSolution);
end

function testInsertingToTail(testCase)
    array = [3 4 5 6 7 8 9 10 0.5];
    array = array(:);
    element = 1;
    k = 10;
    
    actSolution = utils.insertElement(array, element, k);
    
    expSolution = [3 4 5 6 7 8 9 10 0.5 1];
    expSolution = expSolution(:);
    
    verifyEqual(testCase, actSolution, expSolution);
end

function testInsertingToMiddle(testCase)
    array = [3 4 5 6 7 8 9 10 0.5];
    array = array(:);
    element = 1;
    k = 4;
    
    actSolution = utils.insertElement(array, element, k);
    
    expSolution = [3 4 5 1 6 7 8 9 10 0.5];
    expSolution = expSolution(:);
    
    verifyEqual(testCase, actSolution, expSolution);
end

function testCRPCase(testCase)
    K = 8;
    array = [3 4 5 6 7 8 9 10 0.5];
    array = array(:);
    assert((K + 1) == length(array), 'test data is inconsistent with K');
    
    element = 1;
    k = K + 1;
    
    actSolution = utils.insertElement(array, element, k);
    K = K + 1;
    
    expSolution = [3 4 5 6 7 8 9 10 1 0.5];
    expSolution = expSolution(:);
    
    assert((K + 1) == length(actSolution), 'length of array is inconsistent with K');
    verifyEqual(testCase, actSolution, expSolution);
end

function setupOnce(testCase)
end

function teardown(testCase)
end