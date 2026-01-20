function Beltrami_simply()
    
    dataFile = 'mapping_result.mat'; 
    if ~exist(dataFile, 'file')
        error('Data file %s not found.', dataFile);
    end
    load(dataFile, 'XY_Phys', 'UV_Calc', 'tri');

    Nodes_Ref = UV_Calc; 
    Nodes_Phy = XY_Phys;

    mu_list = compute_beltrami(Nodes_Ref, Nodes_Phy, tri);
    mu_abs = abs(mu_list);
    
    mean_mu = mean(mu_abs);
    max_mu = max(mu_abs);
    
    fprintf('Statistics: Max=%.4f, Mean=%.4f\n', max_mu, mean_mu);

    TR = triangulation(tri, Nodes_Phy);
    bdy_edges = freeBoundary(TR);
    bdy_x = [Nodes_Phy(bdy_edges(:,1), 1), Nodes_Phy(bdy_edges(:,2), 1)]';
    bdy_y = [Nodes_Phy(bdy_edges(:,1), 2), Nodes_Phy(bdy_edges(:,2), 2)]';

    figure('Color', 'w', 'Position', [100, 100, 1000, 450]);

    subplot(1, 2, 1);
    trisurf(tri, Nodes_Phy(:,1), Nodes_Phy(:,2), zeros(size(Nodes_Phy,1),1), ...
        'FaceVertexCData', mu_abs, 'FaceColor', 'flat', 'EdgeColor', 'none');
    hold on;
    plot(bdy_x, bdy_y, 'k-', 'LineWidth', 1.5);
    axis equal; axis off; view(2);
    colormap('jet');
    cb = colorbar;
    cb.Label.String = 'Beltrami Coefficient |\mu_{\Phi}|';
    clim([0, prctile(mu_abs, 99)]); 
    title('Spatial Distortion Map');

    subplot(1, 2, 2);
    histogram(mu_abs, 60, 'FaceColor', [0.2, 0.6, 0.8], 'EdgeColor', 'w');
    hold on; grid on; box on;
    xline(mean_mu, 'r--', 'LineWidth', 2.5);
    
    str_stats = {['Mean: ' num2str(mean_mu, '%.4f')], ...
                 ['Max:  ' num2str(max_mu,  '%.4f')]};
    text(0.7, 0.85, str_stats, 'Units', 'normalized', ...
         'FontSize', 12, 'Color', 'r', 'FontWeight', 'bold', ...
         'BackgroundColor', 'w', 'EdgeColor', 'k');

    xlabel('Beltrami Coefficient |\mu_{\Phi}|');
    ylabel('Number of Elements');
    title('Statistical Distribution');
    xlim([0, max(mu_abs)*1.1]);
end

function mu = compute_beltrami(Nodes_Ref, Nodes_Phy, Tri)
    u = Nodes_Ref(:, 1); v = Nodes_Ref(:, 2);
    x = Nodes_Phy(:, 1); y = Nodes_Phy(:, 2);
    
    w = u + 1i*v; 
    z = x + 1i*y; 
    
    n_tri = size(Tri, 1);
    mu = zeros(n_tri, 1);
    
    for k = 1:n_tri
        idx = Tri(k, :);
        w_tri = w(idx); z_tri = z(idx);
        
        a = conj(w_tri(2) - w_tri(3));
        b = conj(w_tri(3) - w_tri(1));
        c = conj(w_tri(1) - w_tri(2));
        denom = w_tri(1)*a + w_tri(2)*b + w_tri(3)*c;
        
        dz_dw = (z_tri(1)*a + z_tri(2)*b + z_tri(3)*c) / denom;
        dz_dw_bar = (z_tri(1)*conj(a) + z_tri(2)*conj(b) + z_tri(3)*conj(c)) / conj(denom);
        
        if abs(dz_dw) < 1e-12
            mu(k) = 0; 
        else
            mu(k) = dz_dw_bar / dz_dw;
        end
    end
end
