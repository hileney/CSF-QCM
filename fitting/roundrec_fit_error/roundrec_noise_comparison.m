clear; close all; clc;

fprintf('1. Initializing geometry and ground truth wavefront...\n');

N_eval = 256; 
x_vec = linspace(-1.4, 1.4, N_eval);
y_vec = linspace(-0.6, 0.6, N_eval);
[X_eval, Y_eval] = meshgrid(x_vec, y_vec);

L = 2.4; H = 0.8; Rad = 0.3;
mask_eval = (abs(X_eval) <= (L/2-Rad) & abs(Y_eval) <= H/2) | ...
            (abs(X_eval) <= L/2 & abs(Y_eval) <= (H/2-Rad)) | ...
            ((abs(X_eval)-(L/2-Rad)).^2 + (abs(Y_eval)-(H/2-Rad)).^2 <= Rad^2 ...
            & abs(X_eval)>(L/2-Rad) & abs(Y_eval)>(H/2-Rad));

z_modes = 36;
rng(2025); 
c2 = zeros(z_modes, 1);
c2(5) = 1.0; c2(7) = 0.5; c2(11:end) = (rand(26,1)-0.5) * 0.1;

[Theta_eval, R_eval] = cart2pol(X_eval, Y_eval);
get_z = @(c, r, t) sum(reshape(cell2mat(arrayfun(@(j) c(j)*zernike_func(j, r, t), ...
    1:z_modes, 'UniformOutput', false)), [size(r), z_modes]), 3);

W_raw = get_z(c2, R_eval, Theta_eval);
[W_true, ~] = force_rms(W_raw, mask_eval, 0.05);

fprintf('2. Preparing sampling points and fitting matrices...\n');

N_bdy = 600;
[xb_ord, yb_ord] = get_rect_boundary(L, H, Rad, N_bdy);
z_boundary = xb_ord + 1i*yb_ord;
scale_factor = max(abs(z_boundary));
z_boundary = z_boundary / scale_factor;

[map_coeffs, ~] = solve_polygon_map(z_boundary, 256);

N_r = 40; N_th = 1024; 
r_vec = sqrt(linspace(0.05^2, 1.0^2, N_r)); 
t_vec = linspace(0, 2*pi, N_th+1); t_vec(end) = [];
[R_c, T_c] = meshgrid(r_vec, t_vec);
Z_phys_conf = polyval_func(map_coeffs, R_c.*exp(1i*T_c)) * scale_factor;

x_conf = real(Z_phys_conf(:)); y_conf = imag(Z_phys_conf(:));
r_conf_c = R_c(:); t_conf_c = T_c(:);

A_conf = zeros(length(x_conf), z_modes);
for j=1:z_modes, A_conf(:,j) = zernike_func(j, r_conf_c, t_conf_c); end

[RR_rec, TT_rec] = meshgrid(linspace(0,1,60), linspace(0,2*pi,200));
ZZ_rec_phys = polyval_func(map_coeffs, RR_rec.*exp(1i*TT_rec)) * scale_factor;
XX_rec_conf = real(ZZ_rec_phys); YY_rec_conf = imag(ZZ_rec_phys);

pts_conf_all = [XX_rec_conf(:), YY_rec_conf(:)];
[~, uniq_idx_conf, ~] = unique(pts_conf_all, 'rows', 'stable');
XX_rec_uniq = pts_conf_all(uniq_idx_conf, 1);
YY_rec_uniq = pts_conf_all(uniq_idx_conf, 2);

if ~exist('mapping_result.mat', 'file')
    error('mapping_result.mat not found! Run CSF Solver first.');
end
load('mapping_result.mat', 'XY_Phys', 'UV_Calc');
x_csf = XY_Phys(:,1); y_csf = XY_Phys(:,2);
[th_csf_c, r_csf_c] = cart2pol(UV_Calc(:,1), UV_Calc(:,2));

valid_csf = r_csf_c <= 1.001;
x_csf = x_csf(valid_csf); y_csf = y_csf(valid_csf);
r_csf_c = r_csf_c(valid_csf); th_csf_c = th_csf_c(valid_csf);

pts_csf_all = [x_csf, y_csf];
[~, uniq_idx_csf, ~] = unique(pts_csf_all, 'rows', 'stable');
x_csf_uniq = pts_csf_all(uniq_idx_csf, 1);
y_csf_uniq = pts_csf_all(uniq_idx_csf, 2);

A_csf = zeros(length(x_csf), z_modes);
for j=1:z_modes, A_csf(:,j) = zernike_func(j, r_csf_c, th_csf_c); end

W_clean_csf = interp2(X_eval, Y_eval, W_true, x_csf, y_csf, 'cubic');

valid_mask = ~isnan(W_true);
F_source = scatteredInterpolant(X_eval(valid_mask), Y_eval(valid_mask), W_true(valid_mask), 'natural', 'linear');
W_clean_conf = F_source(x_conf, y_conf);

fprintf('3. Starting Monte Carlo comparison analysis (N=50)...\n');

noise_levels = linspace(0.05, 0.20, 15); 
n_trials = 50;

res_conf_mean = zeros(length(noise_levels), 1);
res_conf_std  = zeros(length(noise_levels), 1);
res_csf_mean  = zeros(length(noise_levels), 1);
res_csf_std   = zeros(length(noise_levels), 1);

U_ideal_conf = exp(1i * W_clean_conf);
U_ideal_csf  = exp(1i * W_clean_csf);

fprintf('   Progress: ');
for k = 1:length(noise_levels)
    sigma = noise_levels(k);
    
    errs_conf = zeros(n_trials, 1);
    errs_csf  = zeros(n_trials, 1);
    
    if mod(k,2)==0, fprintf('%.0f%% ', k/length(noise_levels)*100); else, fprintf('.'); end
    
    for t = 1:n_trials
        noise_dim = 128; 
        N_real = sigma * randn(noise_dim); 
        N_imag = sigma * randn(noise_dim);
        
        [xg, yg] = meshgrid(linspace(-1.5, 1.5, noise_dim));
        F_noise_r = scatteredInterpolant(xg(:), yg(:), N_real(:), 'natural', 'nearest');
        F_noise_i = scatteredInterpolant(xg(:), yg(:), N_imag(:), 'natural', 'nearest');
        
        nr_conf = F_noise_r(x_conf, y_conf);
        ni_conf = F_noise_i(x_conf, y_conf);
        U_noise_conf = (nr_conf + 1i * ni_conf) / sqrt(2);
        
        W_noisy_conf = W_clean_conf + angle((U_ideal_conf + U_noise_conf) ./ U_ideal_conf);
        
        valid = ~isnan(W_noisy_conf);
        c_hat_conf = A_conf(valid,:) \ W_noisy_conf(valid);
        
        W_vals_rec = get_z(c_hat_conf, RR_rec, TT_rec);
        
        W_vals_rec_flat = W_vals_rec(:);
        F_rec_conf = scatteredInterpolant(XX_rec_uniq, YY_rec_uniq, ...
            W_vals_rec_flat(uniq_idx_conf), 'natural', 'linear');
        
        W_final_conf = F_rec_conf(X_eval, Y_eval);
        
        diff = (W_true - W_final_conf);
        errs_conf(t) = sqrt(nanmean(diff(mask_eval).^2));
        
        nr_csf = F_noise_r(x_csf, y_csf);
        ni_csf = F_noise_i(x_csf, y_csf);
        U_noise_csf = (nr_csf + 1i * ni_csf) / sqrt(2);
        
        W_noisy_csf = W_clean_csf + angle((U_ideal_csf + U_noise_csf) ./ U_ideal_csf);
        
        valid_c = ~isnan(W_noisy_csf);
        c_hat_csf = A_csf(valid_c,:) \ W_noisy_csf(valid_c);
        
        W_vals_csf_nodes = A_csf * c_hat_csf;
        
        F_rec_csf = scatteredInterpolant(x_csf_uniq, y_csf_uniq, ...
            W_vals_csf_nodes(uniq_idx_csf), 'natural', 'linear');
            
        W_final_csf = F_rec_csf(X_eval, Y_eval);
        
        diff = (W_true - W_final_csf);
        errs_csf(t) = sqrt(nanmean(diff(mask_eval).^2));
    end
    
    res_conf_mean(k) = mean(errs_conf);
    res_conf_std(k)  = std(errs_conf);
    res_csf_mean(k)  = mean(errs_csf);
    res_csf_std(k)   = std(errs_csf);
end
fprintf('\nCalculation complete.\n');

figure('Name', 'Robustness Comparison: RoundRect', 'Color', 'w', 'Position', [100, 100, 680, 500]);
hold on; box on; 

noise_pct = noise_levels * 100;

errorbar(noise_pct, res_conf_mean, res_conf_std, 'o-', ...
    'LineWidth', 1.5, 'Color', '#0072BD', ...
    'MarkerFaceColor', '#0072BD', 'CapSize', 8, ...
    'DisplayName', 'Schwarz-Christoffel');

errorbar(noise_pct, res_csf_mean, res_csf_std, 's-', ...
    'LineWidth', 1.5, 'Color', '#D95319', ...
    'MarkerFaceColor', '#D95319', 'CapSize', 8, ...
    'DisplayName', 'CSF-QCM (Proposed)');

grid on;
legend('Location', 'NorthWest', 'FontSize', 11);

xlabel('Input Noise Amplitude (%)', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('Reconstruction RMS Error (\lambda)', 'FontSize', 12, 'FontWeight', 'bold');

title({'Robustness Comparison (Round Rect)', 'Speckle Noise Model (Monte Carlo N=50)'}, ...
    'FontSize', 13);

xlim([4, 21]); 
ylim([0, 0.05]);

save('roundrec_robustness_data.mat', 'noise_levels', 'noise_pct', ...
     'res_conf_mean', 'res_conf_std', ...
     'res_csf_mean', 'res_csf_std');
fprintf('Data saved to roundrec_robustness_data.mat\n');

function [x, y] = get_rect_boundary(L, H, R, N)
    pts_per_unit = N / (2*(L+H)); 
    n_line_x = round((L-2*R) * pts_per_unit);
    n_line_y = round((H-2*R) * pts_per_unit);
    n_arc = round(0.5*pi*R * pts_per_unit);
    t=linspace(0,pi/2,n_arc); x_TR=(L/2-R)+R*cos(t); y_TR=(H/2-R)+R*sin(t);
    t=linspace(pi/2,pi,n_arc); x_TL=(-L/2+R)+R*cos(t); y_TL=(H/2-R)+R*sin(t);
    t=linspace(pi,3*pi/2,n_arc); x_BL=(-L/2+R)+R*cos(t); y_BL=(-H/2+R)+R*sin(t);
    t=linspace(3*pi/2,2*pi,n_arc); x_BR=(L/2-R)+R*cos(t); y_BR=(-H/2+R)+R*sin(t);
    x = [x_TR, linspace(L/2-R,-L/2+R,n_line_x), x_TL, -L/2*ones(1,n_line_y), ...
         x_BL, linspace(-L/2+R,L/2-R,n_line_x), x_BR, L/2*ones(1,n_line_y)];
    y = [y_TR, H/2*ones(1,n_line_x), y_TL, linspace(H/2-R,-H/2+R,n_line_y), ...
         y_BL, -H/2*ones(1,n_line_x), y_BR, linspace(-H/2+R,H/2-R,n_line_y)];
    [x, y] = reparam(x, y, N);
end

function [x_new, y_new] = reparam(x, y, N)
    x = x(:); y = y(:);
    dx = diff(x); dy = diff(y);
    dist_steps = sqrt(dx.^2 + dy.^2);
    keep_mask = [true; dist_steps > 1e-12];
    x_clean = x(keep_mask); y_clean = y(keep_mask);
    dx_clean = diff(x_clean); dy_clean = diff(y_clean);
    d = [0; cumsum(sqrt(dx_clean.^2 + dy_clean.^2))];
    if d(end) < 1e-12, x_new = repmat(x_clean(1), N, 1); y_new = repmat(y_clean(1), N, 1);
    else, d = d / d(end); t_query = linspace(0, 1, N)';
    x_new = interp1(d, x_clean, t_query, 'linear'); y_new = interp1(d, y_clean, t_query, 'linear'); end
end

function [W_out, final_rms] = force_rms(W_in, mask, target_rms)
    vals = W_in(mask); vals = vals - mean(vals);
    scale = target_rms / sqrt(mean(vals.^2));
    W_out = nan(size(W_in)); W_out(mask) = vals * scale;
    final_rms = sqrt(mean((vals*scale).^2));
end

function [coeffs, err] = solve_polygon_map(z_b, N)
    coeffs = zeros(N, 1); coeffs(2) = 1.0; z_b = z_b(:); M = length(z_b);
    ang_t = unwrap(angle(z_b));
    for iter = 1:50
        w = exp(1i * linspace(0, 2*pi, M+1).'); w(end) = [];
        z_c = polyval_func(coeffs, w); 
        ang_c = unwrap(angle(z_c));
        z_p = interp1(ang_t, z_b, ang_c + ang_t(1) - ang_c(1), 'linear', 'extrap');
        coeffs_fft = fft(z_p) / M;
        coeffs_new = zeros(N, 1); n_keep = min(N, M); coeffs_new(1:n_keep) = coeffs_fft(1:n_keep);
        coeffs = coeffs + 0.5 * (coeffs_new - coeffs);
        if norm(z_p - z_c) / sqrt(M) < 1e-4, break; end
    end
    err = norm(z_p - z_c) / sqrt(M);
end

function z = polyval_func(c, w)
    z = zeros(size(w)); ww = ones(size(w));
    for k = 1:length(c), z = z + c(k) * ww; ww = ww .* w; end
end

function Z = zernike_func(j, r, t)
    persistent n_tab m_tab;
    if isempty(n_tab), n_tab=[0 1 1 2 2 2 3 3 3 3 4 4 4 4 4 5 5 5 5 5 5 6 6 6 6 6 6 6 7 7 7 7 7 7 7 7]; 
                       m_tab=[0 1 -1 0 -2 2 -1 1 -3 3 0 -2 2 -4 4 -1 1 -3 3 -5 5 0 -2 2 -4 4 -6 6 -1 1 -3 3 -5 5 -7 7]; end
    if j>length(n_tab), Z=zeros(size(r)); return; end
    n=n_tab(j); m=m_tab(j);
    R=0; for k=0:(n-abs(m))/2, R=R+(-1)^k*factorial(n-k)/(factorial(k)*factorial((n+m)/2-k)*factorial((n-m)/2-k))*r.^(n-2*k); end
    if m>=0, Z=R.*cos(m*t); else, Z=R.*sin(abs(m)*t); end
end
