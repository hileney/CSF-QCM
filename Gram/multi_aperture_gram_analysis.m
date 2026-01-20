function multi_aperture_gram_analysis()
    clear; clc; close all;

    aperture_type = 1; 
    
    fprintf('Step 1: Generating boundary (Type %d)...\n', aperture_type);
    
    N_boundary = 1024; 
    
    switch aperture_type
        case 1 
            fprintf('    -> Generating: Type I (Butterfly/Non-Convex)...\n');
            angles_deg = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360];
            theta_ctrl = deg2rad(angles_deg);
            r_ctrl = [1.0, 0.9, 0.6, 0.25, 0.6, 0.9, 1.0, 0.9, 0.6, 0.25, 0.6, 0.9, 1.0]; 
            
            t_dense = linspace(0, 2*pi, N_boundary+1); t_dense(end) = [];
            r_dense = makima(theta_ctrl, r_ctrl, t_dense);
            x = r_dense .* cos(t_dense);
            y = r_dense .* sin(t_dense);
            
        case 2
            fprintf('    -> Generating: Type II (Rounded Rectangle 3:1)...\n');
            L = 3.0; H = 1.0; R = 0.35; 
            
            perimeter_est = 2*(L-2*R) + 2*(H-2*R) + 2*pi*R;
            pts_per_unit = N_boundary / perimeter_est;
            
            n_line_L = round((L-2*R) * pts_per_unit);
            n_line_H = round((H-2*R) * pts_per_unit);
            n_arc = round(0.5*pi*R * pts_per_unit);
            
            y_R = linspace(-H/2+R, H/2-R, n_line_H); 
            x_R = (L/2) * ones(size(y_R));
            th_tr = linspace(0, pi/2, n_arc); 
            x_TR = (L/2-R) + R*cos(th_tr); y_TR = (H/2-R) + R*sin(th_tr);
            x_T = linspace(L/2-R, -L/2+R, n_line_L);
            y_T = (H/2) * ones(size(x_T));
            th_tl = linspace(pi/2, pi, n_arc); 
            x_TL = (-L/2+R) + R*cos(th_tl); y_TL = (H/2-R) + R*sin(th_tl);
            y_L = linspace(H/2-R, -H/2+R, n_line_H);
            x_L = (-L/2) * ones(size(y_L));
            th_bl = linspace(pi, 3*pi/2, n_arc);
            x_BL = (-L/2+R) + R*cos(th_bl); y_BL = (-H/2+R) + R*sin(th_bl);
            x_B = linspace(-L/2+R, L/2-R, n_line_L);
            y_B = (-H/2) * ones(size(x_B));
            th_br = linspace(3*pi/2, 2*pi, n_arc);
            x_BR = (L/2-R) + R*cos(th_br); y_BR = (-H/2+R) + R*sin(th_br);
            
            x_raw = [x_R, x_TR, x_T, x_TL, x_L, x_BL, x_B, x_BR];
            y_raw = [y_R, y_TR, y_T, y_TL, y_L, y_BL, y_B, y_BR];
            
            [x, y] = reparameterize_curve(x_raw, y_raw, N_boundary);
            
        otherwise
            error('Unknown aperture type: %d', aperture_type);
    end
    
    x = x - mean(x); 
    y = y - mean(y);
    z_boundary = x + 1i*y;
    z_boundary = z_boundary / max(abs(z_boundary)); 
    
    z_boundary = z_boundary(:).';

    fprintf('Step 2: Solving conformal mapping coefficients (Wegmann Iteration)...\n');
    N_coeffs = 256; 
    [coeffs, err_history] = solve_polygon_map(z_boundary, N_coeffs);
    fprintf('    -> Final mapping error: %.2e\n', err_history(end));

    fprintf('Step 3: Generating regular grid and calculating physical weights...\n');
    
    Nr = 120;  
    Nt = 180;  
    
    dr = 1 / Nr;
    dtheta = 2*pi / Nt;
    
    r_vec = linspace(dr/2, 1-dr/2, Nr); 
    theta_vec = linspace(0, 2*pi-dtheta, Nt);
    
    [R_grid, T_grid] = meshgrid(r_vec, theta_vec);
    r_samp = R_grid(:);
    theta_samp = T_grid(:);
    w_samp = r_samp .* exp(1i * theta_samp);
    N_total = length(w_samp);
    
    fprintf('    -> Grid points: %d (Nr=%d, Nt=%d)\n', N_total, Nr, Nt);

    dz_dw = zeros(size(w_samp));
    for j = 2:length(coeffs)
        k = j - 1;
        term = k * coeffs(j) * (w_samp .^ (k - 1));
        dz_dw = dz_dw + term;
    end
    J_val = abs(dz_dw).^2;
    
    Integration_Weights = J_val .* r_samp * dr * dtheta;

    fprintf('Step 4: Calculating Gram matrix (36 Zernike Modes)...\n');
    n_modes = 36;
    Z = zernfun(n_modes, r_samp, theta_samp);

    Total_Physical_Area = sum(Integration_Weights);
    
    G_weighted = (Z' * (Integration_Weights .* Z)) / Total_Physical_Area;
    
    Unit_Disk_Weights = r_samp * dr * dtheta;
    G_ideal = (Z' * (Unit_Disk_Weights .* Z)) / sum(Unit_Disk_Weights);

    cond_ideal = cond(G_ideal);
    cond_real = cond(G_weighted);
    
    fprintf('\n========== Result Analysis (Type %d) ==========\n', aperture_type);
    fprintf('1. Ideal disk Gram condition number: %.4f\n', cond_ideal);
    fprintf('2. Physical domain weighted Gram condition number: %.4f\n', cond_real);
    
    diag_energy = diag(G_weighted);
    fprintf('3. Mode energy (diagonal) range: [%.4f, %.4f]\n', min(diag_energy), max(diag_energy));
    fprintf('========================================\n');

    figure('Position', [100, 100, 1200, 600], 'Color', 'w');
    
    subplot(1, 2, 1);
    z_phys = polyval_func(coeffs, w_samp);
    plot(real(z_phys), imag(z_phys), 'k.', 'MarkerSize', 1); hold on;
    plot(real(polyval_func(coeffs, exp(1i*linspace(0,2*pi,200)))), ...
         imag(polyval_func(coeffs, exp(1i*linspace(0,2*pi,200)))), 'r-', 'LineWidth', 2);
    title({'Mapped Physical Grid', '(Black Dots = Integration Points)'});
    axis equal; axis off;
    
    subplot(1, 2, 2);
    scatter(real(w_samp), imag(w_samp), 10, J_val, 'filled');
    axis equal; colorbar;
    title({'Jacobian (Area Distortion)', 'Unit Disk Domain'});
    xlabel('Re(w)'); ylabel('Im(w)');
    
    figure('Position', [100, 100, 600, 600], 'Color', 'w');
    imagesc(abs(G_weighted));
    colorbar; axis square;
    title(['Cond = ' num2str(cond_real, '%.2f')],FontSize=18);
    xlabel('Zernike Mode Index',FontSize=14); ylabel('Zernike Mode Index',FontSize=14);
    clim([0 1]);

    fprintf('\nStep 6: Wavefront reconstruction verification...\n');
    a_true = zeros(n_modes, 1); a_true(4) = 1.0; a_true(7) = 0.5;
    
    noise_level = 0.01;
    W_measured = Z * a_true + noise_level * randn(size(Z, 1), 1);
    
    b_rhs = (Z' * (Integration_Weights .* W_measured)) / Total_Physical_Area;
    
    lambda = 1e-3;
    a_reg = (G_weighted + lambda * eye(n_modes)) \ b_rhs;
    
    fit_error = norm(a_reg - a_true) / norm(a_true);
    fprintf('    -> Overall fitting relative error: %.2f%%\n', fit_error * 100);

end

function [x_new, y_new] = reparameterize_curve(x, y, N)
    x = x(:).'; y = y(:).';
    
    if norm([x(1)-x(end), y(1)-y(end)]) > 1e-6
        x = [x, x(1)]; y = [y, y(1)];
    end
    
    dx = diff(x); dy = diff(y);
    ds = sqrt(dx.^2 + dy.^2); 
    ds(ds < 1e-12) = 1e-12; 
    
    s = [0, cumsum(ds)];
    total_len = s(end);
    
    t = s / total_len;
    
    [t_unique, idx] = unique(t, 'stable');
    x_unique = x(idx);
    y_unique = y(idx);
    
    t_new = linspace(0, 1, N+1);
    t_new(end) = []; 
    
    x_new = interp1(t_unique, x_unique, t_new, 'pchip');
    y_new = interp1(t_unique, y_unique, t_new, 'pchip');
end

function Z = zernfun(n_modes, r, theta)
    noll_table = [
        0, 0; 1, 1; 1, -1; 2, 0; 2, -2; 2, 2; 3, -1; 3, 1; 3, -3; 3, 3;
        4, 0; 4, 2; 4, -2; 4, 4; 4, -4; 5, 1; 5, -1; 5, 3; 5, -3; 5, 5; 5, -5;
        6, 0; 6, -2; 6, 2; 6, -4; 6, 4; 6, -6; 6, 6;
        7, -1; 7, 1; 7, -3; 7, 3; 7, -5; 7, 5; 7, -7; 7, 7
    ];
    if n_modes > size(noll_table, 1), error('Only supports first 36 modes'); end

    Z = zeros(length(r), n_modes);
    for j = 1:n_modes
        n = noll_table(j, 1); m = noll_table(j, 2);
        m_abs = abs(m);
        R = zeros(size(r));
        k_max = (n - m_abs) / 2;
        for k = 0:k_max
            c = (-1)^k * factorial(n - k) / ...
                (factorial(k) * factorial((n + m_abs)/2 - k) * factorial((n - m_abs)/2 - k));
            R = R + c * r.^(n - 2*k);
        end
        norm_factor = sqrt(n + 1);
        if m ~= 0, norm_factor = sqrt(2 * (n + 1)); end
        
        if m == 0, Z(:, j) = norm_factor * R;
        elseif m > 0, Z(:, j) = norm_factor * R .* cos(m * theta);
        else, Z(:, j) = norm_factor * R .* sin(-m * theta);
        end
        Z(r > 1, j) = 0;
    end
end

function z = polyval_func(coeffs, w)
    z = zeros(size(w));
    ww = ones(size(w));
    for k = 1:length(coeffs)
        z = z + coeffs(k) * ww;
        ww = ww .* w;
    end
end

function [coeffs, err_history] = solve_polygon_map(z_target, N)
    coeffs = zeros(N, 1); coeffs(2) = 1.0; 
    max_iter = 100;
    err_history = zeros(max_iter, 1);
    
    ang_target = angle(z_target);
    ang_target = unwrap(ang_target);
    
    for iter = 1:max_iter
        w_grid = exp(1i * linspace(0, 2*pi, length(z_target)+1));
        w_grid(end) = [];
        z_curr = polyval_func(coeffs, w_grid).'; 
        
        ang_curr = unwrap(angle(z_curr));
        phase_shift = ang_target(1) - ang_curr(1);
        ang_curr_shifted = ang_curr + phase_shift;
        
        z_projected = interp1(ang_target, z_target, ang_curr_shifted, 'linear', 'extrap');
        
        err_history(iter) = norm(z_projected - z_curr);
        if err_history(iter) < 1e-3, err_history = err_history(1:iter); break; end
        
        coeffs_new = fft(z_projected) / length(z_projected);
        coeffs_update = zeros(N, 1);
        n_keep = min(N, length(coeffs_new));
        coeffs_update(1:n_keep) = coeffs_new(1:n_keep);
        coeffs = coeffs + 0.3 * (coeffs_update - coeffs);
    end
end
