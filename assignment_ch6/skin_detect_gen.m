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

% 2: is skin
% 1: is not skin

function [posterior] = skin_detect_gen(x_train, x_test, K)
  
  % Parameters
  dim = 3; % RGB
  I = size(x_train, 1);

  % Classify the training data into two sets. One is skin and
  % the other is not skin.
  x_train_per_class = cell(1, K);
  class_counts = zeros(1, K);
  for k = 1 : K
    idx = find(x_train(:, end) == k);
    x_train_per_class{k} = double(x_train(idx, 1 : end - 1));
    class_counts(k) = size(x_train_per_class{k}, 1);
  end

  % Calculate the mean and covariance matrix of the training data
  % for every class.
  mu = zeros(K, dim);
  cv = cell(1, K);
  lambda = zeros(K, 1);
  for k = 1 : K
    mu(k, :) = mean(x_train_per_class{k});
    cv{k} = cov(x_train_per_class{k});
    lambda(k) = class_counts(k) / I;
  end

  % Compute likelihoods for each observation of testing data.
  x_test = double(x_test);
  likelihoods = zeros(size(x_test, 1), K);
  for k = 1 : K
    likelihoods(:, k) = mvnpdf(x_test, mu(k, :), cv{k});
  end

  % Compute the posterior.
  denominator = 1 ./ (likelihoods * lambda);
  posterior = likelihoods * diag(lambda) .* repmat(denominator, 1, K);

end


