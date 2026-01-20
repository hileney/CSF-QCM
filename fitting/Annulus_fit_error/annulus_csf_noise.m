function annulus_csf_noise()
clear; close all; clc;

fprintf('1. [CSF] Initializing geometry and ground truth wavefront...\n');

N_eval = 200; 
x_vec = linspace(-1.6, 1.6, N_eval);
[X_eval, Y_eval] = meshgrid(x_vec, x_vec);
[Theta_eval, R_eval] = cart2pol(X_eval, Y_eval);

r_out_b = 1.0 + 0.05*sin(3*Theta_eval) + 0.04*cos(5*Theta_eval);
mask_out = R_eval <= r_out_b;

[Theta_in_eval, R_in_local] = cart2pol(X_eval - 0.4, Y_eval + 0.1); 
r_in_b = 0.40 + 0.02*sin(4*Theta_in_eval);
mask_in = R_in_local <= r_in_b;

mask_eval = mask_out & ~mask_in;

z_modes = 36;
rng(2025); 
c_true = (rand(z_modes, 1) - 0.5) ./ (1:z_modes)';
c_true(5:10) = c_true(5:10) * 2; 

W_raw = zeros(size(X_eval));
for j = 1:z_modes
    W_raw = W_raw + c_true(j) * zernike_std(j, R_eval, Theta_eval);
end

vals = W_raw(mask_eval); vals = vals - mean(vals);
scale_factor = 0.05 / std(vals);
W_true = nan(size(W_raw)); 
W_true(mask_eval) = vals * scale_factor;

valid_idx = find(mask_eval);
x_samp = X_eval(valid_idx);
y_samp = Y_eval(valid_idx);
W_clean_samp = W_true(valid_idx);

fprintf('   Sample points: %d\n', length(x_samp));

fprintf('2. [CSF] Generating smooth mesh and building mapping...\n');

N_bdy = 360; 
t = linspace(0, 2*pi, N_bdy+1)'; t(end)=[];

r_out_geo = 1.0 + 0.05*sin(3*t) + 0.04*cos(5*t);
[x_out, y_out] = pol2cart(t, r_out_geo);

r_in_local = 0.40 + 0.02*sin(4*t); 
[x_in_l, y_in_l] = pol2cart(t, r_in_local);
x_in = x_in_l + 0.4; 
y_in = y_in_l - 0.1;

n_ang = 120;
[x_out_u, y_out_u] = reparam_curve(x_out, y_out, n_ang);
[x_in_u, y_in_u] = reparam_curve(x_in, y_in, n_ang);

n_rad = 20; 
r_vals = linspace(0, 1, n_rad)';
X_mesh = zeros(n_rad, n_ang+1); 
Y_mesh = zeros(n_rad, n_ang+1);

for j=1:n_ang+1
    X_mesh(:,j) = (1-r_vals)*x_in_u(j) + r_vals*x_out_u(j);
    Y_mesh(:,j) = (1-r_vals)*y_in_u(j) + r_vals*y_out_u(j);
end

for k=1:40
    X_new = X_mesh; Y_new = Y_mesh;
    for r=2:n_rad-1
        X_new(r,2:end-1) = 0.25*(X_mesh(r-1,2:end-1)+X_mesh(r+1,2:end-1)+X_mesh(r,1:end-2)+X_mesh(r,3:end));
        Y_new(r,2:end-1) = 0.25*(Y_mesh(r-1,2:end-1)+Y_mesh(r+1,2:end-1)+Y_mesh(r,1:end-2)+Y_mesh(r,3:end));
    end
    X_mesh = X_new; Y_mesh = Y_new;
    X_mesh(:,1) = X_mesh(:,end); Y_mesh(:,1) = Y_mesh(:,end);
end

len_out = sum(sqrt(diff(x_out_u).^2 + diff(y_out_u).^2));
len_in  = sum(sqrt(diff(x_in_u).^2  + diff(y_in_u).^2));
epsilon_csf = len_in / len_out;
fprintf('   Estimated Epsilon: %.4f\n', epsilon_csf);

theta_grid = repmat(linspace(0, 2*pi, n_ang+1), n_rad, 1);
rho_grid   = repmat(sqrt(linspace(epsilon_csf^2, 1, n_rad)'), 1, n_ang+1);

F_rho = scatteredInterpolant(X_mesh(:), Y_mesh(:), rho_grid(:), 'natural', 'nearest');
F_sin = scatteredInterpolant(X_mesh(:), Y_mesh(:), sin(theta_grid(:)), 'natural', 'nearest');
F_cos = scatteredInterpolant(X_mesh(:), Y_mesh(:), cos(theta_grid(:)), 'natural', 'nearest');

rho_csf = F_rho(x_samp, y_samp);
sin_csf = F_sin(x_samp, y_samp);
cos_csf = F_cos(x_samp, y_samp);
theta_csf = atan2(sin_csf, cos_csf);

A_csf = zeros(length(x_samp), z_modes);
for j = 1:z_modes
    A_csf(:, j) = zernike_std(j, rho_csf, theta_csf);
end

fprintf('3. [CSF] Starting Monte Carlo noise scan (N=50)...\n');

noise_levels = linspace(0.05, 0.20, 15); 
n_trials = 50;

res_csf_mean = zeros(length(noise_levels), 1);
res_csf_std  = zeros(length(noise_levels), 1);

U_ideal = exp(1i * W_clean_samp); 

fprintf('   Progress: ');
for k = 1:length(noise_levels)
    sigma = noise_levels(k);
    errs_trial = zeros(n_trials, 1);
    
    if mod(k,2)==0, fprintf('%.0f%% ', k/length(noise_levels)*100); else, fprintf('.'); end
    
    for t = 1:n_trials
        n_real = sigma * randn(size(x_samp));
        n_imag = sigma * randn(size(x_samp));
        U_noise = (n_real + 1i * n_imag) / sqrt(2);
        
        U_total = U_ideal + U_noise;
        phase_noise = angle(U_total ./ U_ideal);
        W_noisy = W_clean_samp + phase_noise;
        
        c_csf = A_csf \ W_noisy;
        w_rec = A_csf * c_csf;
        wave_diff = W_clean_samp - w_rec;
        errs_trial(t) = sqrt(mean(wave_diff.^2));
    end
    
    res_csf_mean(k) = mean(errs_trial);
    res_csf_std(k)  = std(errs_trial);
end
fprintf('\nCalculation complete.\n');

save('annulus_csf_noise.mat', 'noise_levels', 'res_csf_mean', 'res_csf_std');
fprintf('Results saved to annulus_csf_noise.mat\n');

end

function [xn, yn] = reparam_curve(x, y, N)
    x = x(:); y = y(:);
    if (x(1) ~= x(end)) || (y(1) ~= y(end)), x(end+1)=x(1); y(end+1)=y(1); end
    dx = diff(x); dy = diff(y); s = [0; cumsum(sqrt(dx.^2 + dy.^2))];
    s_targets = linspace(0, s(end), N+1);
    [s_unique, idx_u] = unique(s);
    xn = interp1(s_unique, x(idx_u), s_targets, 'spline'); 
    yn = interp1(s_unique, y(idx_u), s_targets, 'spline');
end

function Z = zernike_std(j, r, t)
    [n, m] = get_noll_nm(j); R_nl = zeros(size(r));
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
