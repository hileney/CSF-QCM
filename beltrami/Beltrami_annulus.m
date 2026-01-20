function Beltrami_annulus()
    
    dataFile = 'Annulus_mapping.mat';
    if ~exist(dataFile, 'file')
        error('Data file %s not found. Please run Annulus_Mapping_Solver.m or csf_annulus.m first.', dataFile);
    end
    
    fprintf('Loading data from %s...\n', dataFile);
    data = load(dataFile);
    
    if isfield(data, 'X_mesh') && isfield(data, 'Y_mesh') && ~isfield(data, 'XY_Phys')
        fprintf('Detected Mesh format (from csf_annulus.m). Converting to list format...\n');
        
        X_grid = data.X_mesh;
        Y_grid = data.Y_mesh;
        [nr, nc] = size(X_grid);
        
        XY_Phys = [X_grid(:), Y_grid(:)];
        
        tri = generate_grid_triangulation(nr, nc);
        
        R_opt = 0.53374917;
        
        rho_vals = linspace(R_opt, 1, nr);
        theta_vals = linspace(0, 2*pi, nc);
        
        [Theta_grid, Rho_grid] = meshgrid(theta_vals, rho_vals);
        [U_grid, V_grid] = pol2cart(Theta_grid, Rho_grid);
        
        UV_Calc = [U_grid(:), V_grid(:)];
        
        fprintf('  -> Converted Grid: %dx%d\n', nr, nc);
        fprintf('  -> Estimated R_opt (Perimeter Ratio): %.4f\n', R_opt);
        
    else
        if ~isfield(data, 'XY_Phys'), error('Data missing XY_Phys or X_mesh.'); end
        XY_Phys = data.XY_Phys;
        UV_Calc = data.UV_Calc;
        tri     = data.tri;
        R_opt   = data.R_opt;
        fprintf('Loaded Standard Data format.\n');
    end
    
    fprintf('Loaded R_opt = %.6f\n', R_opt);

    fprintf('Computing Beltrami coefficients...\n');
    mu_list = compute_beltrami(UV_Calc, XY_Phys, tri);
    mu_abs = abs(mu_list);
    
    valid_mask = isfinite(mu_abs);
    mu_abs = mu_abs(valid_mask);
    tri_plot = tri(valid_mask, :); 
    mean_mu = mean(mu_abs);
    max_mu = max(mu_abs);
    
    fprintf('Distortion Stats: Max=%.4f, Mean=%.4f\n', max_mu, mean_mu);

    figure('Color', 'w', 'Position', [100, 100, 1100, 500]);
    
    subplot(1, 2, 1);
    trisurf(tri_plot, XY_Phys(:,1), XY_Phys(:,2), zeros(size(XY_Phys,1),1), ...
        'FaceVertexCData', mu_abs, 'FaceColor', 'flat', 'EdgeColor', 'none');
    hold on;
    axis equal; axis off; view(2);
    colormap('jet'); 
    cb = colorbar;
    cb.Label.String = 'Beltrami Coefficient |\mu|';
    cb.Label.FontSize = 12;
    
    if ~isempty(mu_abs)
        clim([0, prctile(mu_abs, 99.5)]); 
    end
    title(['Spatial Distortion Map (R_{opt} \approx ' num2str(R_opt, '%.4f') ')'], 'FontSize', 14);

    subplot(1, 2, 2);
    histogram(mu_abs, 60, 'FaceColor', [0.2, 0.6, 0.8], 'EdgeColor', 'w');
    hold on; grid on; box on;
    
    yl = ylim;
    line([mean_mu, mean_mu], yl, 'Color', 'r', 'LineStyle', '--', 'LineWidth', 2);
    text(mean_mu, yl(2)*0.95, sprintf(' Mean: %.4f', mean_mu), 'Color', 'r', 'FontWeight', 'bold');

    dim = [0.65 0.6 0.2 0.2];
    str = {['R_{opt}: ' num2str(R_opt, '%.6f')], ...
           ['Max |\mu|: ' num2str(max_mu, '%.4f')], ...
           ['Mean |\mu|: ' num2str(mean_mu, '%.4f')]};
    annotation('textbox', dim, 'String', str, 'FitBoxToText', 'on', ...
               'BackgroundColor', 'w', 'EdgeColor', 'k', 'FontSize', 12);

    xlabel('Beltrami Coefficient |\mu|');
    ylabel('Frequency');
    title('Distortion Distribution', 'FontSize', 14);
    if ~isempty(max_mu)
        xlim([0, max(0.1, max_mu*1.1)]); 
    end
end

function tri = generate_grid_triangulation(nr, nc)
    
    num_quads = (nr-1) * (nc-1);
    tri = zeros(num_quads * 2, 3);
    
    cnt = 1;
    for c = 1:nc-1
        for r = 1:nr-1
            p1 = r + (c-1)*nr;     
            p2 = r + 1 + (c-1)*nr; 
            p3 = r + c*nr;         
            p4 = r + 1 + c*nr;     
            
            tri(cnt, :)   = [p1, p2, p3];
            tri(cnt+1, :) = [p2, p4, p3];
            cnt = cnt + 2;
        end
    end
end

function mu = compute_beltrami(Nodes_Ref, Nodes_Phy, Tri)
    w = Nodes_Ref(:,1) + 1i*Nodes_Ref(:,2);
    z = Nodes_Phy(:,1) + 1i*Nodes_Phy(:,2);
    
    idx1 = Tri(:, 1); idx2 = Tri(:, 2); idx3 = Tri(:, 3);
    
    w1 = w(idx1); w2 = w(idx2); w3 = w(idx3);
    z1 = z(idx1); z2 = z(idx2); z3 = z(idx3);
    
    a = conj(w2 - w3);
    b = conj(w3 - w1);
    c = conj(w1 - w2);
    
    denom = w1.*a + w2.*b + w3.*c;
    
    dz_dw = (z1.*a + z2.*b + z3.*c) ./ denom;
    dz_dw_bar = (z1.*conj(a) + z2.*conj(b) + z3.*conj(c)) ./ conj(denom);
    
    mu = nan(size(Tri, 1), 1); 
    
    valid = abs(denom) > 1e-12 & abs(dz_dw) > 1e-12;
    mu(valid) = dz_dw_bar(valid) ./ dz_dw(valid);
end
