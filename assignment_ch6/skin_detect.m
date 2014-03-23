%
% Copyright (C) 2014  Che-Hsun Liu  <chehsunliu@gmail.com>
% 
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see [http://www.gnu.org/licenses/].
%

function skin_detect()

  % Parameters
  image_file    = 'sample1.png';          % 195 x 195 x 3
  label_file    = 'sample1_labeled.png';  % 195 x 195 x 3
  test_file     = 'test1.png';            % 720 x 480 x 3
  test_out_file = 'test1_out.png';        % 720 x 480 x 3

  % Load the training image, labeled image, and testing image.
  % Note that the labeled image is converted to binary. 1 means
  % not skin and 2 means skin. 
  im = imread(image_file);
  im_label = rgb2gray(imread(label_file));
  im_label = (im_label > 128) + 1; % Binarized
  [rows cols] = size(im_label);
  im_test = imread(test_file);
  [rows_test cols_test dump] = size(im_test);

  % Organize the training data and testing data.
  x_train = [reshape(im, rows * cols, 3) reshape(im_label, rows * cols, 1)];
  x_test = reshape(im_test, rows_test * cols_test, 3);

  % Generative model
  posterior = skin_detect_gen(x_train, x_test, 2);
  x_test = (posterior(:, 1) < posterior(:, 2)) * 255;
  x_test = reshape(x_test, rows_test, cols_test);
  figure(1);
  imshow(uint8(x_test));
  imwrite(uint8(x_test), test_out_file);

end


