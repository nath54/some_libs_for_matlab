% Main filtering function
function filtered_img = apply_frequency_filter(img, fn_filter)
    % Convert image to double if not already
    img = double(img);
    
    % Get image dimensions
    [M, N] = size(img);
    
    % Compute center coordinates
    center_x = ceil((M + 1)/2);
    center_y = ceil((N + 1)/2);
    
    % Create coordinate matrices
    [X, Y] = meshgrid(1:N, 1:M);
    
    % Calculate distance from center for each point
    distances = sqrt((X - center_y).^2 + (Y - center_x).^2);
    
    % Create the filter mask by applying fn_filter to distances
    % Normalize distances to [0, 1] range by dividing by max possible distance
    max_distance = sqrt((M/2)^2 + (N/2)^2);
    mask = fn_filter(distances / max_distance);
    
    % Apply FFT
    F = fftshift(fft2(img));
    
    % Apply filter
    F_filtered = F .* mask;
    
    % Inverse FFT and take real part
    filtered_img = real(ifft2(ifftshift(F_filtered)));
    
    % Normalize output to original range
    filtered_img = filtered_img - min(filtered_img(:));
    filtered_img = filtered_img * (max(img(:)) / max(filtered_img(:)));
end

% Linear filter function
% d: normalized distance from center [0, 1]
% cutoff: distance at which filter reaches 0 [0, 1]
function value = linear_filter(d, cutoff)
    if nargin < 2
        cutoff = 0.5; % default cutoff
    end
    
    % Linear decrease from 1 to 0
    value = max(0, 1 - (d / cutoff));
end


% Linear filter function (high-pass)
% d: normalized distance from center [0, 1]
% cutoff: distance at which filter reaches 1 [0, 1]
function value = linear_filter_inverse(d, cutoff)
    if nargin < 2
        cutoff = 0.5; % default cutoff
    end
    
    % Linear increase from 0 to 1
    value = min(1, d / cutoff);
end

% Gaussian filter function
% d: normalized distance from center [0, 1]
% sigma: standard deviation of the Gaussian [0, 1]
function value = gaussian_filter(d, sigma)
    if nargin < 2
        sigma = 0.2; % default sigma
    end
    
    % Gaussian function
    value = exp(-(d.^2) / (2*sigma^2));
end


clear;


% Load an image
img = imread("/MATLAB Drive/TraitementDImage/imagesTest/house.jpg");
if size(img, 3) > 1
    img = rgb2gray(img); % Convert to grayscale if needed
end

% Create filter function handles with specific parameters
linear_filter_handle = @(d) linear_filter(d, 0.5);  % 0.5 is the cutoff
inv_linear_filter_handle = @(d) linear_filter_inverse(d, 0.5);  % 0.5 is the cutoff
gaussian_filter_handle = @(d) gaussian_filter(d, 0.2);  % 0.2 is sigma


% Apply filters
linear_filtered = apply_frequency_filter(img, linear_filter_handle);
inv_linear_filtered = apply_frequency_filter(img, inv_linear_filter_handle);
gaussian_filtered = apply_frequency_filter(img, gaussian_filter_handle);

% Display results
figure;
subplot(1,4,1); imshow(img); title('Original');
subplot(1,4,2); imshow(uint8(linear_filtered)); title('Linear Filter');
subplot(1,4,3); imshow(uint8(inv_linear_filtered)); title('Inv Linear Filter');
subplot(1,4,4); imshow(uint8(gaussian_filtered)); title('Gaussian Filter');