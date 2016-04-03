load('santander.mat');
data = trainData;
labels = trainLabels;

cleanData = zscore(data);
cleanTest = zscore(testData);

N = size(cleanData,1);

% remove constant columns
M = size(cleanData,2);
keep = 1:M;
for i = 1:M
    if std(cleanData(:,i)) == 0
        keep(keep==i) = [];
    end
end
cleanData = cleanData(:, keep);
cleanTest = cleanTest(:,keep);

% remove duplicated colums
M = size(cleanData,2);
keep = 1:M;
for i = 1:M
    v = cleanData(:,i);
    for j = i+1:M
        if sum(v ~= cleanData(:,j)) == 0
            keep(keep==j) = [];
        end
    end
end
cleanData = cleanData(:, keep);
cleanTest = cleanTest(:,keep);


% oversample the minority class
[adaFeats, adaLabels] = ADASYN(cleanData, labels, 1);
cleanData = [cleanData; adaFeats];
labels = [labels; adaLabels];


%% svd plotting
% 
% [U,S,V] = svd(cleanData, 'econ');
% plot3(U(sat,1), U(sat,2), U(sat,3), 'k.');
% hold on
% plot3(U(dis,1), U(dis,2), U(dis,3), 'r.');
% hold off;
%% pick a validation set
N = size(cleanData,1);
M = size(cleanData,2);

sat = labels==1;
dis = labels==0;
disp(sum(sat));
disp(sum(dis));

satInds = find(sat);
disInds = find(dis);

satFrac = sum(sat)/N;
disFrac = sum(dis)/N;

valN = N*0.1;
valSatN = ceil(valN*satFrac);
valDisN = valSatN; floor(valN*disFrac);

satInds = find(sat);
disInds = find(dis);

valSatInds = randperm(length(satInds), valSatN);
valDisInds = randperm(length(disInds), valDisN);

valInds = [satInds(valSatInds); disInds(valDisInds)];

satInds(valSatInds) = [];
disInds(valDisInds) = [];

trainInds = [satInds; disInds];

targets = zeros(2,N);
for i = 1:N
    targets(labels(i)+1,i)=1;
end
%% train the neural network

net = patternnet([256, 128, 64, 32, 16]);
net.divideFcn = 'divideind';
net.divideParam.trainInd = trainInds;
net.divideParam.valInd = valInds;
% net.divideParam.testInd = [];

[net,tr] = train(net, cleanData', targets);
plotperform(tr)
outs = net(cleanTest');
preds = outs(2,:)';
%% predict on test set and output results to submission file
fID = fopen('submission.csv','w');
fprintf(fID, 'ID,TARGET\n');

for i = 1:size(preds,1)
    fprintf(fID, '%i,%d\n', testID(i), preds(i));
end

fclose(fID);




