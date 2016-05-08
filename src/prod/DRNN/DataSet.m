classdef DataSet < handle
    properties (Abstract = true)
        index, dataset
    end
    
    methods(Abstract = true)
        shuffle(object)
        setNextIndex(object)
        getFeature(obj)
        getLabel(obj)
    end
    
    methods (Static = true)
        function vec = label2vec(label, classNum, T)
            batchSize = length(label);
            vec = zeros(classNum, batchSize, T);
            
            for i=1:batchSize
                vec(label(i),i,:) = 1;
            end
        end
    end
end