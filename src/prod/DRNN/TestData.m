classdef TestData < DataSet
    properties
        index, batchSize, N, dataset
        dataMat, labels
    end
    
    methods
        function obj = TestData(batchSize, dataset)
            obj.batchSize = batchSize;
            obj.index = (1:batchSize) - batchSize;
            obj.dataset = dataset;
        end
        
        function [trainSet, validSet, testSet] = stratify(obj, dataName)
            load(dataName);
            
            trainSet = TestData(obj.batchSize, 'training');
            trainSet.setData(trainMat);
            trainSet.setLabel(trainLabel');
            
            validSet = TestData(obj.batchSize, 'validation');
            validSet.N = 0;
            
            testSet = TestData(obj.batchSize, 'test');
            testSet.setData(testMat);
            testSet.setLabel(testLabel');
        end
        
        function setData(obj, dataMat)
            obj.dataMat = dataMat;
            obj.N = size(dataMat,2);
        end
        
        function setLabel(obj, labels)
            obj.labels = labels;
        end
        
        function feature = getFeature(obj)
            feature = obj.dataMat(:,obj.index,:);
        end
        
        function label = getLabel(obj)
            label = obj.labels(obj.index);
        end
        
        function shuffle(obj)
            rndidx = randperm(obj.N);
            
            obj.dataMat = obj.dataMat(:,rndidx,:);
            obj.labels = obj.labels(rndidx);
        end
        
        function goNext = setNextIndex(obj)
            obj.index = obj.index + obj.batchSize;
            
            goNext = true;
            
            if obj.index(end) > obj.N
                obj.index = (1:obj.batchSize) - obj.batchSize;
                goNext = false;
            end
        end
    end
end