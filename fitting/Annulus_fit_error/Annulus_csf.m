clear; clc; close all;

fprintf('[Step 1] Generating Ground Truth Wavefront...\n');

N = 400;           
target_rms = 0.05; 
rng(2025);

x = linspace(-1.6, 1.6, N);
y = linspace(-1.6, 1.6, N);
[X, Y] = meshgrid(x, y);
[Theta, R] = cart2pol(X, Y);

r_out_b = 1.0 + 0.05*sin(3*Theta) + 0.04*cos(5*Theta);
mask_out = R <= r_out_b;
[Theta_in, R_in_local] = cart2pol(X - 0.4, Y + 0.1); 
r_in_b = 0.40 + 0.02*sin(4*Theta_in);
mask_in = R_in_local <= r_in_b;
mask = mask_out & ~mask_in;

z_modes = 36;
coeffs_true = (rand(z_modes, 1) - 0.5) .* (1./(1:z_modes)');
coeffs_true(5:10) = coeffs_true(5:10) * 2; 
W_raw = zeros(size(X));
for j = 1:z_modes
    W_raw = W_raw + coeffs_true(j) * zernike_std(j, R, Theta);
end

vals = W_raw(mask);
vals = vals - mean(vals);
W_true = nan(size(W_raw));
W_true(mask) = vals * (target_rms / std(vals));
rms_true = std(W_true(mask));
fprintf('   Ground Truth RMS: %.4f lambda\n', rms_true);

fprintf('[Step 2] Generating CSF-QCM Mesh (Smoothing Method)...\n');

N_bdy = 360; 
t = linspace(0, 2*pi, N_bdy+1)'; t(end) = []; 

r_out_geo = 1.0 + 0.05*sin(3*t) + 0.04*cos(5*t);
[x_out, y_out] = pol2cart(t, r_out_geo);

r_in_local = 0.40 + 0.02*sin(4*t); 
[x_in_local, y_in_local] = pol2cart(t, r_in_local);
x_in = x_in_local + 0.40; 
y_in = y_in_local - 0.10;

n_angular = 120;
n_radial = 20;
[x_out_u, y_out_u] = reparam_curve(x_out, y_out, n_angular);
[x_in_u, y_in_u] = reparam_curve(x_in, y_in, n_angular);

r_vals = linspace(0, 1, n_radial)';
X_mesh = zeros(n_radial, n_angular+1);
Y_mesh = zeros(n_radial, n_angular+1);

for j = 1:n_angular+1
    P_in = [x_in_u(j), y_in_u(j)];
    P_out = [x_out_u(j), y_out_u(j)];
    X_mesh(:, j) = (1-r_vals) * P_in(1) + r_vals * P_out(1);
    Y_mesh(:, j) = (1-r_vals) * P_in(2) + r_vals * P_out(2);
end

fprintf('   Executing Laplacian Smoothing...\n');
n_iter = 50;
for k = 1:n_iter
    X_new = X_mesh; Y_new = Y_mesh;
    for r = 2:n_radial-1
        for c = 1:n_angular+1
            c_prev = c - 1; if c_prev < 1, c_prev = n_angular; end
            c_next = c + 1; if c_next > n_angular+1, c_next = 2; end
            
            X_new(r,c) = 0.25 * (X_mesh(r-1,c) + X_mesh(r+1,c) + X_mesh(r,c_prev) + X_mesh(r,c_next));
            Y_new(r,c) = 0.25 * (Y_mesh(r-1,c) + Y_mesh(r+1,c) + Y_mesh(r,c_prev) + Y_mesh(r,c_next));
        end
    end
    X_mesh = X_new; Y_mesh = Y_new;
end

fprintf('[Step 3] Mapping Pixels to CSF Canonical Domain...\n');

x_out_col = x_out_u(:); y_out_col = y_out_u(:);
x_in_col  = x_in_u(:);  y_in_col  = y_in_u(:);

d_out = diff([x_out_col; x_out_col(1)]).^2 + diff([y_out_col; y_out_col(1)]).^2;
len_out = sum(sqrt(d_out));

d_in = diff([x_in_col; x_in_col(1)]).^2 + diff([y_in_col; y_in_col(1)]).^2;
len_in = sum(sqrt(d_in));

epsilon_csf = len_in / len_out;
fprintf('   Estimated Epsilon (Perimeter Ratio): %.4f\n', epsilon_csf);

theta_mesh = repmat(linspace(0, 2*pi, n_angular+1), n_radial, 1);
rho_sq_vals = linspace(epsilon_csf^2, 1, n_radial)';
rho_vals = sqrt(rho_sq_vals);
rho_mesh = repmat(rho_vals, 1, n_angular+1);

Sin_mesh = sin(theta_mesh);
Cos_mesh = cos(theta_mesh);

F_rho = scatteredInterpolant(X_mesh(:), Y_mesh(:), rho_mesh(:), 'natural', 'nearest');
F_sin = scatteredInterpolant(X_mesh(:), Y_mesh(:), Sin_mesh(:), 'natural', 'nearest');
F_cos = scatteredInterpolant(X_mesh(:), Y_mesh(:), Cos_mesh(:), 'natural', 'nearest');

fprintf('[Step 4] Fitting Wavefront on CSF Grid...\n');

valid_indices = find(mask);
x_pix = X(valid_indices);
y_pix = Y(valid_indices);
w_pix_raw = W_true(valid_indices);

rho_pix = F_rho(x_pix, y_pix);
sin_pix = F_sin(x_pix, y_pix);
cos_pix = F_cos(x_pix, y_pix);
theta_pix = atan2(sin_pix, cos_pix);

nan_filter = isnan(rho_pix) | isnan(theta_pix) | isnan(w_pix_raw);
rho_fit = rho_pix(~nan_filter);
theta_fit = theta_pix(~nan_filter);
w_fit = w_pix_raw(~nan_filter);
final_idx = valid_indices(~nan_filter);

rho_fit = max(epsilon_csf, min(1, rho_fit));

n_terms = 36;
H_csf = zeros(length(rho_fit), n_terms);
for j = 1:n_terms
    H_csf(:, j) = zernike_std(j, rho_fit, theta_fit); 
end
[Q_csf, ~] = qr(H_csf, 0);
coeffs_csf = Q_csf' * w_fit;
w_rec_vec = Q_csf * coeffs_csf;

W_rec = nan(size(W_true)); W_rec(final_idx) = w_rec_vec;
Resid = nan(size(W_true)); Resid(final_idx) = w_fit - w_rec_vec;

rms_resid = sqrt(mean((w_fit - w_rec_vec).^2));
fprintf('   Fitting Result (CSF-Mesh): RMS Resid = %.5f lambda\n', rms_resid);

pv_true=max(W_true(:))-min(W_true(:));
fprintf('   Ture PV: %.5f λ \n', pv_true);

pv_conf=max(Resid(:))-min(Resid(:));
fprintf('   CSF Fit PV: %.5f λ \n', pv_conf);

figure('Name','CSF Fitting Comparison',...
       'Color','w',...
       'Position',[100 100 1400 420]);

t = tiledlayout(1,3,...
    'TileSpacing','compact',...
    'Padding','compact');

Mask_Plot = double(mask);
Mask_Plot(~mask) = NaN;

cm_main   = jet(256);
clim_main = [-0.16 0.16];
clim_err  = clim_main / 5;

ax_gt = nexttile(t,1);
pcolor(X,Y,W_true .* Mask_Plot);
shading interp;
axis image off;

colormap(ax_gt, cm_main);
clim(ax_gt, clim_main);
title('Ground Truth');

ax_rec = nexttile(t,2);
pcolor(X,Y,W_rec .* Mask_Plot);
shading interp;
axis image off;

colormap(ax_rec, cm_main);
clim(ax_rec, clim_main);
title('CSF Reconstruction');

ax_err = nexttile(t,3);
pcolor(X,Y,Resid .* Mask_Plot);
shading interp;
axis image off;

colormap(ax_err, cm_main);
clim(ax_err, clim_err);
title({'Residual Error',...
       sprintf('RMS = %.5f \\lambda', rms_resid)});

cb_main = colorbar(ax_rec,'Location','eastoutside');
cb_main.Label.String = 'Wavefront / \lambda';

pos_rec = ax_rec.Position;
pos_gt  = ax_gt.Position;

cb_main.Position = [ ...
    pos_rec(1) + pos_rec(3) + 0.012, ...
    pos_gt(2), ...
    0.018, ...
    pos_gt(4) ];

cb_err = colorbar(ax_err,'Location','eastoutside');
cb_err.Label.String = 'Residual / \lambda';

pos_err = ax_err.Position;
cb_err.Position = [ ...
    pos_err(1) + pos_err(3) + 0.012, ...
    pos_err(2), ...
    0.018, ...
    pos_err(4) ];

function [xn, yn] = reparam_curve(x, y, N)
    x = x(:); y = y(:);
    if (x(1) ~= x(end)) || (y(1) ~= y(end)), x(end+1)=x(1); y(end+1)=y(1); end
    dx = diff(x); dy = diff(y);
    dist = sqrt(dx.^2 + dy.^2);
    s = [0; cumsum(dist)];
    s_targets = linspace(0, s(end), N+1);
    xn = interp1(s, x, s_targets, 'spline'); 
    yn = interp1(s, y, s_targets, 'spline');
end

function Z = zernike_std(j, r, t)
    [n, m] = get_noll_nm(j);
    R_nl = zeros(size(r));
    for k = 0:(n-abs(m))/2
        num = (-1)^k * factorial(n-k);
        den = factorial(k) * factorial((n+m)/2 - k) * factorial((n-m)/2 - k);
        R_nl = R_nl + (num/den) * r.^(n-2*k);
    end
    if m >= 0, Z = R_nl .* cos(m*t); else, Z = R_nl .* sin(abs(m)*t); end
end

function [n, m] = get_noll_nm(j)
    n_list = [0 1 1 2 2 2 3 3 3 3 4 4 4 4 4 5 5 5 5 5 5 6 6 6 6 6 6 6 7 7 7 7 7 7 7 7];
    m_list = [0 1 -1 0 -2 2 -1 1 -3 3 0 -2 2 -4 4 -1 1 -3 3 -5 5 0 -2 2 -4 4 -6 6 -1 1 -3 3 -5 5 -7 7];
    if j > length(n_list), n=0; m=0; else, n=n_list(j); m=m_list(j); end
end
