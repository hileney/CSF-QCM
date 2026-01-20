clear; clc; close all;

fprintf('Initializing Type II rounded rectangle boundary...\n');
L = 2.4; H = 0.8; R = 0.3; 
N_pts = 600;

pts_per_unit = N_pts / (2*(L+H)); 
n_line_x = round((L-2*R) * pts_per_unit);
n_line_y = round((H-2*R) * pts_per_unit);
n_arc = round(0.5*pi*R * pts_per_unit);

y_R = linspace(-H/2+R, H/2-R, n_line_y); x_R = (L/2) * ones(size(y_R));
t = linspace(0, pi/2, n_arc); 
x_TR = (L/2-R) + R*cos(t); y_TR = (H/2-R) + R*sin(t);
x_T = linspace(L/2-R, -L/2+R, n_line_x); y_T = (H/2) * ones(size(x_T));
t = linspace(pi/2, pi, n_arc); 
x_TL = (-L/2+R) + R*cos(t); y_TL = (H/2-R) + R*sin(t);
y_L = linspace(H/2-R, -H/2+R, n_line_y); x_L = (-L/2) * ones(size(y_L));
t = linspace(pi, 3*pi/2, n_arc); 
x_BL = (-L/2+R) + R*cos(t); y_BL = (-H/2+R) + R*sin(t);
x_B = linspace(-L/2+R, L/2-R, n_line_x); y_B = (-H/2) * ones(size(x_B));
t = linspace(3*pi/2, 2*pi, n_arc); 
x_BR = (L/2-R) + R*cos(t); y_BR = (-H/2+R) + R*sin(t);

clear; close all; clc;

fprintf('1. Initializing Type II (L=2.4, H=0.8) geometry and wavefront...\n');

N_eval = 512;
x_vec = linspace(-1.4, 1.4, N_eval);
y_vec = linspace(-0.6, 0.6, N_eval);
[X_eval, Y_eval] = meshgrid(x_vec, y_vec);

L = 2.4; H = 0.8; Rad = 0.3;
mask_eval = (abs(X_eval) <= (L/2-Rad) & abs(Y_eval) <= H/2) | ...
            (abs(X_eval) <= L/2 & abs(Y_eval) <= (H/2-Rad)) | ...
            ((abs(X_eval)-(L/2-Rad)).^2 + (abs(Y_eval)-(H/2-Rad)).^2 <= Rad^2 ...
            & abs(X_eval)>(L/2-Rad) & abs(Y_eval)>(H/2-Rad));

N_bdy = 600;
[xb_ord, yb_ord] = get_rect_boundary(L, H, Rad, N_bdy);
z_boundary = xb_ord + 1i*yb_ord;
scale_factor = max(abs(z_boundary));
z_boundary = z_boundary / scale_factor;

z_modes = 36;
rng(2025);
dummy = rand(36, 1); 

c2 = zeros(z_modes, 1);
c2(5) = 1.0; 
c2(7) = 0.5;
c2(11:end) = (rand(26,1)-0.5) * 0.1;

[Theta_eval, R_eval] = cart2pol(X_eval, Y_eval);
get_z = @(c, r, t) sum(reshape(cell2mat(arrayfun(@(j) c(j)*zernike_func(j, r, t), ...
    1:z_modes, 'UniformOutput', false)), [size(r), z_modes]), 3);

R_norm = R_eval ; 
W_raw = get_z(c2, R_norm, Theta_eval);

target_rms = 0.05;
[W_true, true_rms] = force_rms(W_raw, mask_eval, target_rms);

fprintf('   Ground truth wavefront generated (RMS=%.4f λ)\n', true_rms);

fprintf('2. Running Conformal Mapping (modified sampling strategy)...\n');

N_coeffs = 256;
[map_coeffs, ~] = solve_polygon_map(z_boundary, N_coeffs);

N_r = 40; 
N_th = 1024; 

r_vec = sqrt(linspace(0.05^2, 1.0^2, N_r)); 
t_vec = linspace(0, 2*pi, N_th+1); t_vec(end) = [];
[R_c, T_c] = meshgrid(r_vec, t_vec);
W_comp = R_c .* exp(1i * T_c);

Z_phys = polyval_func(map_coeffs, W_comp) * scale_factor;
x_conf = real(Z_phys(:)); 
y_conf = imag(Z_phys(:));
r_conf_c = R_c(:); t_conf_c = T_c(:);

valid_mask = ~isnan(W_true);
x_src = X_eval(valid_mask);
y_src = Y_eval(valid_mask);
v_src = W_true(valid_mask);

F_source = scatteredInterpolant(x_src, y_src, v_src, 'natural', 'linear');

W_meas_conf = F_source(x_conf, y_conf);
valid = ~isnan(W_meas_conf);
x_conf_fit = x_conf(valid); y_conf_fit = y_conf(valid);
r_conf_fit = r_conf_c(valid); t_conf_fit = t_conf_c(valid);
W_meas_fit = W_meas_conf(valid);

A_conf = zeros(length(W_meas_fit), z_modes);
for j=1:z_modes, A_conf(:,j) = zernike_func(j, r_conf_fit, t_conf_fit); end
c_fit_conf = A_conf \ W_meas_fit;

[RR_rec, TT_rec] = meshgrid(linspace(0,1,100), linspace(0,2*pi,512));
ZZ_rec_phys = polyval_func(map_coeffs, RR_rec.*exp(1i*TT_rec)) * scale_factor;
W_rec_vals = get_z(c_fit_conf, RR_rec, TT_rec); 

F_conf = scatteredInterpolant(real(ZZ_rec_phys(:)), imag(ZZ_rec_phys(:)), W_rec_vals(:), 'natural', 'linear');

W_rec_conf = F_conf(X_eval, Y_eval);

W_rec_conf(~mask_eval) = NaN; 

Res_conf = W_true - W_rec_conf;
Res_conf(~mask_eval) = NaN; 
rms_conf = sqrt(nanmean(Res_conf(:).^2));

pv_conf=max(Res_conf(:))-min(Res_conf(:));

fprintf('   Conformal RMS Error: %.5f λ \n', rms_conf);
fprintf('   Conformal RMS Error: %.5f λ \n', pv_conf);

pv_true=max(W_true(:))-min(W_true(:));

fprintf('   Ture PV: %.5f λ \n', pv_true);

fprintf('3. Loading CSF-QCM data (mapping_result.mat)...\n');
if ~exist('mapping_result.mat', 'file')
    warning('mapping_result.mat not found! Run CSF Solver first. Skipping CSF part.');
    has_csf = false;
else
    has_csf = true;
    load('mapping_result.mat', 'XY_Phys', 'UV_Calc');
    
    x_csf = XY_Phys(:,1); y_csf = XY_Phys(:,2);
    u_csf = UV_Calc(:,1); v_csf = UV_Calc(:,2);
    [th_csf_c, r_csf_c] = cart2pol(u_csf, v_csf);
    
    W_meas_csf = interp2(X_eval, Y_eval, W_true, x_csf, y_csf, 'cubic');
    valid = ~isnan(W_meas_csf) & r_csf_c <= 1.001; 
    x_csf = x_csf(valid); y_csf = y_csf(valid);
    r_csf_c = r_csf_c(valid); th_csf_c = th_csf_c(valid);
    W_meas_csf = W_meas_csf(valid);
    
    A_csf = zeros(length(W_meas_csf), z_modes);
    for j=1:z_modes, A_csf(:,j) = zernike_func(j, r_csf_c, th_csf_c); end
    c_fit_csf = A_csf \ W_meas_csf;
    
    W_vals_csf = A_csf * c_fit_csf; 
    F_csf = scatteredInterpolant(x_csf, y_csf, W_vals_csf, 'natural', 'linear');
    W_rec_csf = F_csf(X_eval, Y_eval);
    Res_csf = W_true - W_rec_csf;
    rms_csf = sqrt(nanmean(Res_csf(:).^2));

    pv_csf=max(Res_csf(:))-min(Res_csf(:));
    
    fprintf('   CSF-QCM RMS Error:   %.5f λ\n', rms_csf);
    fprintf('   CSF-QCM PV Error:   %.5f λ\n', pv_csf);
end

climit = [-0.15, 0.15]; 
plot_res = 400; 

x_lim = max(abs(x_conf(:)));
y_lim = max(abs(y_conf(:)));
R_geom = y_lim;          
L_rect = x_lim - R_geom; 

xv_plot = linspace(-x_lim, x_lim, plot_res);
yv_plot = linspace(-y_lim, y_lim, plot_res);
[X_plot, Y_plot] = meshgrid(xv_plot, yv_plot);

Mask_Smooth = (abs(X_plot) <= L_rect & abs(Y_plot) <= R_geom) | ...
              ((abs(X_plot) > L_rect) & ((abs(X_plot)-L_rect).^2 + Y_plot.^2 <= R_geom^2));
Mask_Val = double(Mask_Smooth);
Mask_Val(~Mask_Smooth) = NaN; 

figure('Name', '3. Ground Truth Wavefront', 'Color', 'w');

pcolor(X_eval, Y_eval, W_true .* double(mask_eval)); 
shading interp;
axis image; axis off;
colormap(jet);
clim(climit);
colorbar;
title('Ground Truth (Type II)', 'FontSize', 12);

figure('Name', '4. Conformal Reconstruction', 'Color', 'w');
h1 = pcolor(X_eval, Y_eval, W_rec_conf); 
set(h1, 'EdgeColor', 'none');
shading interp;
axis image; axis off;
colormap(jet);
clim(climit); 
colorbar;
title('Conformal Recon', 'FontSize', 14);

figure('Name', '5. Conformal Residual', 'Color', 'w');
h2 = pcolor(X_eval, Y_eval, Res_conf);
set(h2, 'EdgeColor', 'none');
shading interp;
axis image; axis off;
colormap(jet);
clim([-0.055, 0.055]); 
cb = colorbar;
ylabel(cb, '\lambda');
title({'Conformal Resid', ['RMS = ' num2str(rms_conf, '%.5f') '\lambda']}, 'FontSize', 14);

if exist('has_csf', 'var') && has_csf
    
    Mask_CSF_Val = double(mask_eval);
    Mask_CSF_Val(mask_eval == 0) = NaN; 
    
    figure('Name', '6. CSF Reconstruction', 'Color', 'w');
    pcolor(X_eval, Y_eval, W_rec_csf .* Mask_CSF_Val); 
    shading interp; 
    axis image; axis off;
    colormap(jet); 
    clim(climit); 
    colorbar;
    title('CSF Recon', 'FontSize', 12);
    
    figure('Name', '7. CSF Residual', 'Color', 'w');
    pcolor(X_eval, Y_eval, Res_csf .* Mask_CSF_Val); 
    shading interp; 
    axis image; axis off;
    colormap(jet); 
    clim(climit/2); 
    colorbar;
    title({'CSF Resid', ['RMS=' num2str(rms_csf, '%.4f') '\lambda']}, 'FontSize', 12);
end

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
    
    dx = diff(x);
    dy = diff(y);
    dist_steps = sqrt(dx.^2 + dy.^2);
    
    keep_mask = [true; dist_steps > 1e-12];
    x_clean = x(keep_mask);
    y_clean = y(keep_mask);
    
    dx_clean = diff(x_clean);
    dy_clean = diff(y_clean);
    d = [0; cumsum(sqrt(dx_clean.^2 + dy_clean.^2))];
    
    if d(end) < 1e-12
        x_new = repmat(x_clean(1), N, 1);
        y_new = repmat(y_clean(1), N, 1);
    else
        d = d / d(end);
        t_query = linspace(0, 1, N)';
        x_new = interp1(d, x_clean, t_query, 'linear'); 
        y_new = interp1(d, y_clean, t_query, 'linear');
    end
end

function [W_out, final_rms] = force_rms(W_in, mask, target_rms)
    vals = W_in(mask); vals = vals - mean(vals);
    scale = target_rms / sqrt(mean(vals.^2));
    W_out = nan(size(W_in)); W_out(mask) = vals * scale;
    final_rms = sqrt(mean((vals*scale).^2));
end

function [coeffs, err] = solve_polygon_map(z_b, N)
    coeffs = zeros(N, 1); 
    coeffs(2) = 1.0; 
    
    z_b = z_b(:);
    M = length(z_b); 
    
    ang_t = unwrap(angle(z_b));
    
    max_iter = 50;
    for iter = 1:max_iter
        w = exp(1i * linspace(0, 2*pi, M+1).'); 
        w(end) = []; 
        
        z_c = polyval_func(coeffs, w); 
        
        ang_c = unwrap(angle(z_c));
        
        phase_shift = ang_t(1) - ang_c(1);
        
        z_p = interp1(ang_t, z_b, ang_c + phase_shift, 'linear', 'extrap');
        
        coeffs_fft = fft(z_p) / M;
        
        coeffs_new = zeros(N, 1);
        n_keep = min(N, M);
        coeffs_new(1:n_keep) = coeffs_fft(1:n_keep);
        
        coeffs = coeffs + 0.5 * (coeffs_new - coeffs);
        
        err = norm(z_p - z_c) / sqrt(M);
        if err < 1e-4
            break; 
        end
    end
end

function z = polyval_func(c, w)
    z = zeros(size(w));
    ww = ones(size(w));
    for k = 1:length(c)
        z = z + c(k) * ww;
        ww = ww .* w;
    end
end

function Z = zernike_func(j, r, t)
    persistent n_tab m_tab;
    if isempty(n_tab), n_tab=[0 1 1 2 2 2 3 3 3 3 4 4 4 4 4]; m_tab=[0 1 -1 0 -2 2 -1 1 -3 3 0 -2 2 -4 4]; end
    if j>length(n_tab), Z=zeros(size(r)); return; end
    n=n_tab(j); m=m_tab(j);
    R=0; for k=0:(n-abs(m))/2, R=R+(-1)^k*factorial(n-k)/(factorial(k)*factorial((n+m)/2-k)*factorial((n-m)/2-k))*r.^(n-2*k); end
    if m>=0, Z=R.*cos(m*t); else, Z=R.*sin(abs(m)*t); end
end
