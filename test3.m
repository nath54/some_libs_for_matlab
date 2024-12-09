
% Load image
image = imread('imagesTest/house.jpg');
image = rgb2gray(image); % Convert to grayscale

% Downsample and upsample functions
function downsampled = downsample2(img)
    downsampled = img(1:2:end, 1:2:end);
end

function upsampled = upsample2(img)
    upsampled = imresize(img, 2, 'bilinear'); % Bilinear interpolation
end

% Visualize original, downsampled, and upsampled images
figure;
subplot(1, 3, 1);
title('Original Image');
imshow(image, []);
subplot(1, 3, 2);
title('Downsampled');
imshow(downsample2(image), []);
subplot(1, 3, 3);
title('Upsampled');
imshow(upsample2(downsample2(image)), []);

% Gaussian blur and reduction
function reduced = reduce2(img)
    blurred = imgaussfilt(img, 1); % Apply Gaussian blur
    reduced = downsample2(blurred);
end

% Gaussian pyramid
function pyramid_layers = pyramid(img, levels)
    pyramid_layers = cell(1, levels);
    pyramid_layers{1} = img;
    for i = 2:levels
        img = reduce2(img);
        pyramid_layers{i} = img;
    end
end

% Visualize Gaussian Pyramid
pyr = pyramid(image, 4);
figure;
for i = 1:length(pyr)
    subplot(2, 2, i);
    title(['Level ', num2str(i)]);
    imshow(pyr{i}, []);
end

% Expand with Gaussian blur
function expanded = expand2(img)
    upsampled = upsample2(img);
    expanded = imgaussfilt(upsampled, 1); % Apply Gaussian blur
end

% Laplacian pyramid
function laplacian_pyr = laplacian_pyramid(img, levels)
    gaussian_pyr = pyramid(img, levels);
    laplacian_pyr = cell(1, levels);
    for i = 1:levels-1
        expanded = expand2(gaussian_pyr{i+1});
        % Ensure sizes match
        [rows, cols] = size(gaussian_pyr{i});
        expanded = expanded(1:rows, 1:cols);
        laplacian_pyr{i} = gaussian_pyr{i} - expanded;
    end
    laplacian_pyr{levels} = gaussian_pyr{end}; % Add the smallest level
end

% Visualize Laplacian Pyramid
lap_pyr = laplacian_pyramid(image, 4);
figure;
for i = 1:length(lap_pyr)
    subplot(2, 2, i);
    title(['Level ', num2str(i)]);
    imshow(lap_pyr{i}, []);
end
