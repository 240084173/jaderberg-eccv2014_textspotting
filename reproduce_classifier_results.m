% Max Jaderberg 2014
% Reproduce classifier results (Table 2)

addpath(genpath('./'));

dset = {};
model = {};
classchans = {};

%% Case-insensitive
dset{end+1} = 'data/icdar2003-chars-test.mat';
model{end+1} = 'models/charnet_layers.mat';
classchans{end+1} = 2:37;  % ignore background class

%% Case-sensitive
dset{end+1} = 'data/icdar2003-charscases-test.mat';
model{end+1} = 'models/casesnet_layers.mat';
classchans{end+1} = 3:64;  % ignore background classes

%% Bigrams IC03 only
dset{end+1} = 'data/icdar2003-bigramsic03-test.mat';
model{end+1} = 'models/bigramic03net_layers.mat';
classchans{end+1} = 2:400;  % ignore background class

assert(numel(dset) == numel(model));
for i=1:numel(dset)
    fprintf('Testing %s ...\n', model{i});
    % load model
    nn = cudaconvnet_to_mconvnet(model{i});
    % load data
    s = load(dset{i});
    ims = [];
    labels = [];
    for j=1:numel(s.gt.labels)
        ims = cat(4, ims, s.gt.images{j}{:});
        labels = cat(2, labels, (j)*ones(1,numel(s.gt.images{j})));
    end
    ims = single(ims);
    labels = single(labels);
    % preprocess
    data = reshape(ims, [], size(ims,4));
    mu = mean(data, 1);
    data = data - repmat(mu, size(data,1), 1);
    v = std(data, 0, 1);
    data = data ./ (0.0000001 + repmat(v, size(data,1), 1));
    ims = reshape(data, size(ims));
    clear data;
    
    nn = nn.forward(nn, struct('data', single(ims)));
    
    %% go
    [~,pred] = max(squeeze(nn.Xout(:,:,classchans{i},:)), [], 1);
    err = sum(labels == pred) / numel(pred);
    fprintf('\taccuracy: %.2f percent\n', err*100);
end