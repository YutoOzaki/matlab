classdef GTZAN < DataSet
    properties
        pathList, labels
        index, batchSize, N, dataset
        AFE
    end
    
    methods(Static = true)
        function pathList = readFilePath(filePath)
            fileID = fopen(filePath);
            path = textscan(fileID, '%s');
            fclose(fileID);

            pathList = path{1,1};
        end
    end
    
    methods
        function obj = GTZAN(pathList, batchSize, dataset)
            obj.pathList = pathList;
            obj.N = length(obj.pathList);
            createLabels(obj);
            
            obj.batchSize = batchSize;
            
            obj.index = (1:obj.batchSize) - obj.batchSize;
            obj.dataset = dataset;
        end
        
        function createLabels(obj)
            keySet =   {'blues', 'classical', 'country', 'disco','hiphop','jazz','metal','pop','reggae','rock'};
            valueSet = [1 2 3 4 5 6 7 8 9 10];
            mapObj = containers.Map(keySet,valueSet);
            
            obj.labels = zeros(obj.N,1);
            
            for i=1:obj.N
                label_str = strsplit(strrep(obj.pathList{i},'\','/'), '/');
                obj.labels(i) = mapObj(label_str{end-1});
            end
        end
        
        function [gtzan_train, gtzan_valid, gtzan_test] = stratify(obj, N_training, N_validation, N_testing)
            stratifiedFilePath = {cell(N_training, 1), cell(N_validation, 1), cell(N_testing, 1)};
            
            N_ = [N_training N_validation N_testing];
            
            total = sum(N_);
            assert(total == obj.N, 'Total number of samples is not agreed\n');
            
            classLabel = unique(obj.labels);
            classNum = length(classLabel);
            
            r_class = zeros(classNum, 1);
            label_idx = cell(classNum, 1);
            for i=1:classNum
                label_idx{i} = find(obj.labels == classLabel(i));
                r_class(i) = length(label_idx{i});
            end
            r_class = r_class./total;
            
            for i=1:3
                N_class = floor(r_class .* N_(i));
                k = 1;
                
                for j=1:classNum
                   label_j_index = label_idx{j};
                   rndidx = randperm(length(label_j_index), N_class(j));
                   stratifiedFilePath{i}(k:k+N_class(j)-1) = obj.pathList(label_j_index(rndidx));
                   
                   label_idx{j} = setdiff(label_idx{j}, label_j_index(rndidx));
                   k = k + N_class(j);
                end
            end
            
            remaining = cell2mat(label_idx);
            if isempty(remaining) == false
                remaining = remaining(randperm(length(remaining)));
                
                for i=1:3
                    checkEmpty = cellfun(@isempty, stratifiedFilePath{i});
                    emptyIdx = find(checkEmpty == 1);
                    stratifiedFilePath{i}(emptyIdx) = obj.pathList(remaining(1:length(emptyIdx)));
                    
                    remaining = remaining(length(emptyIdx)+1:end);
                end
            end
            
            gtzan_train = GTZAN(stratifiedFilePath{1}, obj.batchSize, 'training');
            gtzan_valid = GTZAN(stratifiedFilePath{2}, obj.batchSize, 'validation');
            gtzan_test = GTZAN(stratifiedFilePath{3}, obj.batchSize, 'test');
        end
        
        function setAFE(obj, type)
            if strcmp(type, 'MFCC')
                obj.AFE = MFCC();
            end
        end
        
        function shuffle(obj)
            rndidx = randperm(obj.N);
            
            obj.pathList = obj.pathList(rndidx);
            obj.labels = obj.labels(rndidx);
        end
        
        function feature = getFeature(obj)
            cellFormat = obj.AFE.extract(obj.pathList(obj.index));
            feature = batchFormat(obj, cellFormat);
        end
        
        function feature = batchFormat(obj, cellFormat)
            dim = cellfun(@(x) size(x,1), cellFormat);
            T = cellfun(@(x) size(x,2), cellFormat);
            T_min = min(T);
            
            feature = zeros(dim(1), obj.batchSize, T_min);
            for i=1:obj.batchSize
                feature(:,i,:) = cellFormat{i}(:,1:T_min);
            end
        end
        
        function label = getLabel(obj)
            label = obj.labels(obj.index);
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