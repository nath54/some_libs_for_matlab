function img_normalized = normalize_image_to_256(img)
    % Normalize image pixel values around the mean and scale to [0, 255].

    % Convert to double
    img = double(img);

    % Calculate mean
    mean_val = mean(img(:));

    % Shift values around the mean (centering at 128)
    img_normalized = (img - mean_val) + 128;

    % Normalize to [0, 255]
    img_normalized = (img_normalized - min(img_normalized(:))) / ...
                     (max(img_normalized(:)) - min(img_normalized(:))) * 255;

    % Clip to [0, 255] and cast to uint8
    img_normalized = uint8(max(0, min(255, img_normalized)));
end


function extracted_channel = extract_color_channel(img, channel)
    % Extract specific color channel from an RGB image.

    if size(img, 3) ~= 3
        error('Input image must be an RGB image.');
    end

    if channel < 1 || channel > 3
        error('Channel must be 1 (Red), 2 (Green), or 3 (Blue).');
    end

    % Extract the channel
    extracted_channel = img(:, :, channel);
end


function rgb_image = combine_grayscale_to_rgb(red_channel, green_channel, blue_channel)
    % Combine grayscale images into an RGB image.

    if ~isequal(size(red_channel), size(green_channel), size(blue_channel))
        error('All channels must have the same dimensions.');
    end

    % Normalize if necessary
    red_channel = normalize_image_to_256(red_channel);
    green_channel = normalize_image_to_256(green_channel);
    blue_channel = normalize_image_to_256(blue_channel);

    % Combine into RGB
    rgb_image = cat(3, red_channel, green_channel, blue_channel);
end


function transposed_img = transpose_image(img)
    % Transpose the dimensions of an image.

    if ndims(img) == 2
        % Grayscale image
        transposed_img = img';
    elseif ndims(img) == 3
        % RGB image
        transposed_img = permute(img, [2, 1, 3]);
    else
        error('Unsupported image format.');
    end
end


function filtered_img = median_filter(img, block_size)
    % Apply median filter to a grayscale image.

    if nargin < 2
        block_size = 3; % Default block size
    end

    % Pad the image
    pad_size = floor(block_size / 2);
    padded_img = padarray(img, [pad_size, pad_size], 0, 'both');

    % Create the filtered image
    [rows, cols] = size(img);
    filtered_img = zeros(rows, cols);

    for i = 1:rows
        for j = 1:cols
            % Extract block
            block = padded_img(i:i+block_size-1, j:j+block_size-1);
            
            % Compute median
            filtered_img(i, j) = median(block(:));
        end
    end

    filtered_img = uint8(filtered_img);
end


function filtered_img = median_rgb_filter(img, block_size)
    % Apply median filter to an RGB image.

    if nargin < 2
        block_size = 3; % Default block size
    end

    if size(img, 3) ~= 3
        error('Input image must be an RGB image with 3 channels.');
    end

    % Pad the image
    pad_size = floor(block_size / 2);
    padded_img = padarray(img, [pad_size, pad_size, 0], 0, 'both');

    % Initialize output
    [rows, cols, ~] = size(img);
    filtered_img = zeros(rows, cols, 3, 'uint8');

    for i = 1:rows
        for j = 1:cols
            % Extract block for each channel
            block = padded_img(i:i+block_size-1, j:j+block_size-1, :);

            % Compute median for each channel
            for channel = 1:3
                filtered_img(i, j, channel) = median(block(:, :, channel), 'all');
            end
        end
    end
end







% Load an RGB image
img = imread("/MATLAB Drive/TraitementDImage/imagesTest/house.jpg");

% Normalize the image
normalized_img = normalize_image_to_256(img);

% Extract the red channel
red_channel = extract_color_channel(img, 1);

% Combine channels into RGB
rgb_img = combine_grayscale_to_rgb(red_channel, red_channel, red_channel);

% Transpose the image
transposed_img = transpose_image(img);

% Apply median filter
filtered_img_gray = median_filter(rgb2gray(img), 5);
filtered_img_rgb = median_rgb_filter(img, 5);

% Display results
imshow(normalized_img);
figure; imshow(filtered_img_rgb);
figure; imshow(transposed_img);
figure; imshow(rgb_img);
figure; imshow(filtered_img_gray);

