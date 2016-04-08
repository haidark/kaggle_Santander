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


% svd plotting
% 
% [U,S,V] = svd(cleanData, 'econ');
% plot3(U(sat,1), U(sat,2), U(sat,3), 'k.');
% hold on
% plot3(U(dis,1), U(dis,2), U(dis,3), 'r.');
% hold off;

%% train H models on different validation splits of the data
clc
H = 10;
outs = zeros(H, size(cleanTest,1));
models = cell(H,1);
parfor h = 1:H
    % pick a validation set
    N = size(cleanData,1);
    M = size(cleanData,2);

    sat = labels==1;
    dis = labels==0;

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
    % train the neural network
    layout = [randi(512,1) randi(256,1) randi(128,1) randi(64) randi(32) randi(16)];
    disp(['Training model #{' num2str(h) '}'])
    disp(['Model #{' num2str(h) '} hidden layers: ' num2str(layout)])
    break
    net = patternnet(layout);
    net.divideFcn = 'divideind';
    net.divideParam.trainInd = trainInds;
    net.divideParam.valInd = valInds;

    [net,tr] = train(net, cleanData', targets);
    models{h} = net;
%     plotperform(tr)
    out = net(cleanTest');
    outs(h,:) = out(2,:);
    disp(['Done training model #{' num2str(h) '}...']);
end

%% output results to submission file
preds = mean(outs,1)';
fID = fopen('submission.csv','w');
fprintf(fID, 'ID,TARGET\n');

for i = 1:size(preds,1)
    fprintf(fID, '%i,%d\n', testID(i), preds(i));
end

fclose(fID);





