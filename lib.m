% Libs


% Normalize image pixel values around the mean and scale to [0, 255].
function img_normalized = normalize_image_to_256(img)

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


% Extract specific color channel from an RGB image.
function extracted_channel = extract_color_channel(img, channel)

    if size(img, 3) ~= 3
        error('Input image must be an RGB image.');
    end

    if channel < 1 || channel > 3
        error('Channel must be 1 (Red), 2 (Green), or 3 (Blue).');
    end

    % Extract the channel
    extracted_channel = img(:, :, channel);
end


% Combine grayscale images into an RGB image.
function rgb_image = combine_grayscale_to_rgb(red_channel, green_channel, blue_channel)

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


% Transpose the dimensions of an image.
function transposed_img = transpose_image(img)

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


% Apply median filter to a grayscale image.
function filtered_img = median_filter(img, block_size)

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


% Apply median filter to an RGB image.
function filtered_img = median_rgb_filter(img, block_size)

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


% Generate RGB colormap
function res = rgb_colormap(img, scale)
    img = imresize(img, scale, 'Antialiasing', true);
    [h, w, ~] = size(img);
    reshaped_img = reshape(img, [], 3);
    [unique_colors, ~, idx] = unique(reshaped_img, 'rows');
    counts = histcounts(idx, size(unique_colors, 1));
    log_counts = ceil(log(counts + 1)); % Avoid log(0)
    tot_colors = sum(log_counts);
    rw = floor(tot_colors / 20);
    res = zeros(rw, tot_colors, 3, 'uint8');
    current_height = 1;
    for i = 1:size(unique_colors, 1)
        color_block = repmat(reshape(unique_colors(i, :), 1, 1, 3), rw, log_counts(i));
        res(:, current_height:current_height + log_counts(i) - 1, :) = color_block;
        current_height = current_height + log_counts(i);
    end
    res = transpose_image(res);
end


% Show image with histograms
function show_image(img, title_str, normalize, colorbar_flag, histogram_flag)
    if nargin < 3, normalize = true; end
    if nargin < 4, colorbar_flag = true; end
    if nargin < 5, histogram_flag = true; end
    if normalize
        img = normalize_image_to_256(img);
    end
    figure;
    if size(img, 3) == 1
        subplot(1, 2, 1);
        imshow(img, [0, 255]);
        title(title_str);
        if histogram_flag
            subplot(1, 2, 2);
            imhist(img, 256);
            title('Grayscale Histogram');
        end
    else
        subplot(1, 2, 1);
        imshow(img);
        title(title_str);
        if histogram_flag
            subplot(1, 3, 2); imhist(img(:, :, 1), 256);
            subplot(1, 3, 3); imhist(img(:, :, 2), 256);
            subplot(1, 3, 1); imhist(img(:, :, 3), 256);
            legend({'Red', 'Green', 'Blue'});
            title('RGB Histograms');
        end
    end
end


% Grayscale to RGB conversion
function grayscale = rgb_to_grayscale(img)
    grayscale = uint8(mean(img, 3));
end


% Checkerboard pattern
function C = checkerboard(dim, tile_size, inversed)
    if nargin < 3, inversed = false; end
    [X, Y] = meshgrid(1:dim, 1:dim);
    t = mod(floor(X/tile_size) + floor(Y/tile_size) + inversed, 2) * 255;
    C = uint8(t);
end


% Generate sinusoid 1
function value = sinusoide1(x, y, f1)
    if nargin < 3, f1 = 0.24; end
    value = (1 + cos(2 * pi * f1 * x)) / 2 * 255;
end


% Generate sinusoid 2
function value = sinusoide2(x, y, f1, cx, cy)
    if nargin < 3, f1 = 0.24; end
    if nargin < 4, cx = 128; cy = 128; end
    d = sqrt((x - cx).^2 + (y - cy).^2);
    value = (1 + cos(2 * pi * f1 * d)) / 2 * 255;
end


% Generate grayscale image by function
function img = img_grayscale_created_by_function(dim_x, dim_y, func_handle, fn_kargs)
    if nargin < 4, fn_kargs = struct; end
    [X, Y] = meshgrid(1:dim_x, 1:dim_y);
    img = func_handle(X, Y, fn_kargs);
    img = uint8(img);
end



%
function result = threshold_filter(v, threshold_value, inequality_type, else_value)
    if nargin < 3
        inequality_type = 1;
    end
    if nargin < 4
        else_value = 0;
    end
    d = v - threshold_value;
    if d * inequality_type > 0
        result = v;
    else
        result = else_value;
    end
end


%
function output = npconvolve2d(image, kernel)
    if ndims(image) == 3
        % If RGB, convolve each channel separately
        output = zeros(size(image));
        for i = 1:3
            output(:,:,i) = npconvolve2d(image(:,:,i), kernel);
        end
        return;
    end
    
    % Pad the image
    pad_size = floor(size(kernel) / 2);
    padded_image = padarray(image, pad_size, 'replicate');
    
    % Initialize output
    output = zeros(size(image));
    
    % Perform convolution
    for x = 1:size(image, 2)
        for y = 1:size(image, 1)
            region = padded_image(y:y+size(kernel, 1)-1, x:x+size(kernel, 2)-1);
            output(y, x) = sum(region .* kernel, 'all');
        end
    end
end


%
function kernel = average_kernel(size)
    if mod(size, 2) == 0
        error('Kernel size must be an odd integer.');
    end
    kernel = ones(size, size) / (size * size);
end


%
function kernel = binomial_kernel(size)
    if mod(size, 2) == 0
        error('Kernel size must be an odd integer.');
    end
    kernel = zeros(size, size);
    for i = 0:size-1
        for j = 0:size-1
            kernel(i+1, j+1) = nchoosek(size-1, i) * nchoosek(size-1, j);
        end
    end
    kernel = kernel / sum(kernel, 'all');
end


%
function kernel = gaussian_kernel(size, sigma)
    if nargin < 2
        sigma = 1.0;
    end
    [x, y] = meshgrid(-floor(size/2):floor(size/2), -floor(size/2):floor(size/2));
    kernel = exp(-(x.^2 + y.^2) / (2 * sigma^2));
    kernel = kernel / sum(kernel, 'all');
end


%
function output = gaussian_blur(image, kernel_size, sigma)
    if nargin < 2
        kernel_size = 5;
    end
    if nargin < 3
        sigma = 1.0;
    end
    kernel = gaussian_kernel(kernel_size, sigma);
    output = npconvolve2d(image, kernel);
end


%
function edge_image = sobel_edge_detection(image)
    if ndims(image) == 3
        image = rgb_to_grayscale(image);
    end
    
    kernel_x = [-1 0 1; -2 0 2; -1 0 1];
    kernel_y = [1 2 1; 0 0 0; -1 -2 -1];
    
    grad_x = npconvolve2d(image, kernel_x);
    grad_y = npconvolve2d(image, kernel_y);
    
    gradient_magnitude = sqrt(grad_x.^2 + grad_y.^2);
    edge_image = uint8(gradient_magnitude / max(gradient_magnitude(:)) * 255);
end


%
function edge_image = roberts_edge_detection(image)
    if ndims(image) == 3
        image = rgb_to_grayscale(image);
    end
    
    kernel_x = [1 0; 0 -1];
    kernel_y = [0 1; -1 0];
    
    grad_x = npconvolve2d(image, kernel_x);
    grad_y = npconvolve2d(image, kernel_y);
    
    gradient_magnitude = sqrt(grad_x.^2 + grad_y.^2);
    edge_image = uint8(gradient_magnitude / max(gradient_magnitude(:)) * 255);
end


%
function edge_image = prewitt_edge_detection(image)
    if ndims(image) == 3
        image = rgb_to_grayscale(image);
    end
    
    kernel_x = [-1 0 1; -1 0 1; -1 0 1];
    kernel_y = [-1 -1 -1; 0 0 0; 1 1 1];
    
    grad_x = npconvolve2d(image, kernel_x);
    grad_y = npconvolve2d(image, kernel_y);
    
    gradient_magnitude = sqrt(grad_x.^2 + grad_y.^2);
    edge_image = uint8(gradient_magnitude / max(gradient_magnitude(:)) * 255);
end


%
function edge_image = laplacian_edge_detection(image)
    if ndims(image) == 3
        image = rgb_to_grayscale(image);
    end
    
    kernel = [0 1 0; 1 -4 1; 0 1 0];
    
    laplacian = npconvolve2d(image, kernel);
    edge_image = uint8(abs(laplacian) / max(abs(laplacian(:))) * 255);
end


%
function normalized = min_max_normalization(img, new_min, new_max)
    if nargin < 2, new_min = 0.0; end
    if nargin < 3, new_max = 255.0; end

    img_min = min(img(:));
    img_max = max(img(:));
    normalized = (img - img_min) / (img_max - img_min);
    normalized = normalized * (new_max - new_min) + new_min;
end


%
function normalized = z_score_normalization(img)
    mean_val = mean(img(:));
    std_val = std(img(:));
    normalized = (img - mean_val) / std_val;
end


%
function equalized = histogram_equalization(img)
    if ndims(img) > 2
        error('Histogram equalization works only on grayscale images.');
    end

    hist_vals = histcounts(img(:), 256, 'BinLimits', [0, 256]);
    cdf = cumsum(hist_vals);
    cdf_normalized = cdf / max(cdf);

    equalized = interp1(linspace(0, 256, length(cdf)), cdf_normalized, img, 'linear', 'extrap');
    equalized = uint8(equalized * 255);
end


%
function img_equalized = adaptive_histogram_equalization(img, tile_size)
    if nargin < 2, tile_size = 8; end
    if ndims(img) > 2
        error('Adaptive histogram equalization works only on grayscale images.');
    end

    img_equalized = zeros(size(img));
    for i = 1:tile_size:size(img, 1)
        for j = 1:tile_size:size(img, 2)
            tile = img(i:min(i+tile_size-1, size(img, 1)), j:min(j+tile_size-1, size(img, 2)));
            tile_equalized = histogram_equalization(tile);
            img_equalized(i:min(i+tile_size-1, size(img, 1)), j:min(j+tile_size-1, size(img, 2))) = tile_equalized;
        end
    end
end


%
function stretched = contrast_stretching(img, low_percentile, high_percentile)
    if nargin < 2, low_percentile = 2; end
    if nargin < 3, high_percentile = 98; end

    low = prctile(img(:), low_percentile);
    high = prctile(img(:), high_percentile);
    stretched = (img - low) * 255.0 / (high - low);
    stretched = uint8(max(min(stretched, 255), 0));
end


%
function noisy_image = add_gaussian_noise(img, mean_val, stddev, noise_intensity, clamp_256)
    if nargin < 2, mean_val = 0; end
    if nargin < 3, stddev = 0.1; end
    if nargin < 4, noise_intensity = 1.0; end
    if nargin < 5, clamp_256 = true; end

    noise = stddev * randn(size(img)) + mean_val;
    noisy_image = img + noise * noise_intensity;

    if clamp_256
        noisy_image = normalize_image_to_256(noisy_image);
    end
end


%
function noisy_image = add_speckle_noise(img, mean_val, stddev, noise_intensity, clamp_256)
    if nargin < 2, mean_val = 0; end
    if nargin < 3, stddev = 0.1; end
    if nargin < 4, noise_intensity = 1.0; end
    if nargin < 5, clamp_256 = true; end

    noise = stddev * randn(size(img)) + mean_val;
    noisy_image = img + img .* noise * noise_intensity;

    if clamp_256
        noisy_image = normalize_image_to_256(noisy_image);
    end
end


%
function noisy_image = add_salt_and_pepper_noise(img, salt_prob, pepper_prob, clamp_256)
    if nargin < 2, salt_prob = 0.01; end
    if nargin < 3, pepper_prob = 0.01; end
    if nargin < 4, clamp_256 = true; end

    noisy_image = img;
    salt_mask = rand(size(img)) < salt_prob;
    pepper_mask = rand(size(img)) < pepper_prob;

    noisy_image(salt_mask) = 255;
    noisy_image(pepper_mask) = 0;

    if clamp_256
        noisy_image = max(min(noisy_image, 255), 0);
    end
end


%
function psnr_val = calculate_psnr(img1, img2, max_pixel_value)
    if nargin < 3, max_pixel_value = 255.0; end

    mse = mean((double(img1) - double(img2)).^2, 'all');
    if mse == 0
        psnr_val = inf;
    else
        psnr_val = 20 * log10(max_pixel_value / sqrt(mse));
    end
end


%
function snr = calculate_snr(signal, noisy_signal)
    signal_variance = var(signal(:));
    noise_variance = var(noisy_signal(:) - signal(:));
    snr = signal_variance / noise_variance;
end


%
function snr_db = calculate_snr_db(signal, noisy_signal)
    signal_variance = var(signal(:));
    noise_variance = var(noisy_signal(:) - signal(:));
    snr_db = 10 * log10(signal_variance / noise_variance);
end



% Image division function
function result = img_division(a, b)
    if b == 0
        result = 255;
    else
        result = a / b;
    end
end



% Mean Square Error function
function result = MSE(Y, YH)
    Y = double(Y);
    YH = double(YH);
    result = mean((Y - YH).^2, 'all');
end



% Peak Signal to Noise Ratio function
function result = PSNR(original, noisy, peak)
    if nargin < 3
        peak = 100;
    end
    mse = mean((original - noisy).^2, 'all');
    result = 10 * log10(peak^2 / mse);
end



% Get window function
function window = get_window(img, x, y, N)
    if nargin < 4
        N = 25;
    end
    
    [h, w, c] = size(img);
    arm = floor(N/2);
    window = zeros(N, N, c);
    
    xmin = max(1, x-arm);  % MATLAB uses 1-based indexing
    xmax = min(w, x+arm+1);
    ymin = max(1, y-arm);
    ymax = min(h, y+arm+1);
    
    % Adjusting indices for MATLAB's 1-based indexing
    window(arm - (y-ymin) + 1:arm + (ymax-y), ...
          arm - (x-xmin) + 1:arm + (xmax-x), :) = ...
          img(ymin:ymax, xmin:xmax, :);
end



% Non-Local Means function
function output = NL_means(img, h, f, t)
    if nargin < 2
        h = 8.5;
    end
    if nargin < 3
        f = 4;
    end
    if nargin < 4
        t = 11;
    end
    
    % neighbourhood size 2f+1
    N = 2*f + 1;
    
    % sliding window size 2t+1
    S = 2*t + 1;
    
    % Filtering Parameter
    sigma_h = h;
    
    % Padding the image
    pad_img = padarray(img, [t+f t+f]);
    
    % Getting the height and width of the image
    [h, w] = size(img);
    [h_pad, w_pad] = size(pad_img);
    
    neigh_mat = zeros(h+S-1, w+S-1, N, N);
    
    % Making a dp neighbourhood for all pixels
    for y = 1:h+S-1
        for x = 1:w+S-1
            neigh_mat(y, x, :, :) = get_window(pad_img, x+f, y+f, 2*f+1);
        end
    end
    
    % Empty image to be filled by the algorithm
    output = zeros(size(img));
    
    % Iterating for each pixel
    fprintf('Processing pixels:\n');
    progress = 0;
    total_pixels = (h-1)*(w-1);
    
    for Y = 1:h
        for X = 1:w
            % Update progress
            progress = progress + 1;
            if mod(progress, 1000) == 0
                fprintf('Progress: %.1f%%\n', 100*progress/total_pixels);
            end
            
            % Shifting for padding
            x = X + t;
            y = Y + t;
            
            % Getting neighbourhood in chunks of search window
            neigh_mat_reshaped = reshape(neigh_mat, h+S-1, w+S-1, N*N);
            a = get_window(neigh_mat_reshaped, x, y, S);
            
            % Getting self Neighbourhood
            b = reshape(neigh_mat(y, x, :, :), [], 1);
            
            % Getting distance of vectorized neighbourhood
            c = bsxfun(@minus, a, b');
            
            % Determining weights
            d = c.^2;
            e = sqrt(sum(d, 3));
            F = exp(-e/(sigma_h^2));
            
            % Summing weights
            Z = sum(F, 'all');
            
            % Calculating average pixel value
            im_part = get_window(pad_img, x+f, y+f, S);
            NL = sum(F.*im_part, 'all');
            output(Y, X) = NL/Z;
        end
    end
end


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

