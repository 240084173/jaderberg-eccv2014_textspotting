% Max Jaderberg 2014
% Produces text/no-text saliency map

addpath(genpath('./'));

imfn = 'data/PICTs0057.JPG';
im = imread(imfn);
img = single(rgb2gray(im));

%% pad
img = padarray(img, [11 11]);

%% preprocess
winsz = 24;
mu = (1/winsz^2) * conv2(img, ones(winsz, winsz, 'single'), 'same');
x_ = (img - mu).^2;
stdim = sqrt((1/winsz^2) * conv2(x_, ones(winsz, winsz, 'single'), 'same'));
data = img - mu;
eps = 1;
data = data ./ (stdim + eps);

%% load model
nn = cudaconvnet_to_mconvnet('models/detnet_layers.mat');

%% process
nn = nn.forward(nn, struct('data', data));

%% fig
close all;
fntsz = 20;
figure(1); 
subplot(1,2,1); imshow(im);
subplot(1,2,2);
imshowc(max(nn.Xout(:,:,2:end), [], 3)); 
figure(2); plot(1:length(nn.time), nn.time);
title('Layer timings');
xlabel('Layer #');
ylabel('Time (s)');
