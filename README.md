# CSF-QCM
minimum dataset for CSF-QCM
# CSF-QCM: Wavefront Fitting via CSF-Guided Progressive Quasi-Conformal Mapping

## 1. Project Overview
 This repository contains the MATLAB implementation for the research paper: **"Wavefront fitting over arbitrary freeform apertures via CSF-guided progressive quasi-conformal mapping"**.

The framework addresses the numerical instability and loss of orthogonality in Zernike fitting on irregular freeform apertures.  It integrates **Curve Shortening Flow (CSF)** with topology-aware quasi-conformal mapping to establish a stable diffeomorphic correspondence between physical apertures and canonical domains (Unit Disk or Annulus).

## 2. Functional Modules & File Mapping

The codebase is organized into **five distinct functional modules** corresponding to the analytical pipeline described in the paper.

### Module 1: Aperture Mesh Generation
**Goal:** Generate the discrete physical domains $\Omega_{phy}$ (triangular meshes) for testing.
*  **Paper Correspondence:** Section 3 (Benchmark Aperture Geometries).
* **Key Files:**
    * `aperture_three.m`: **Master Generation Script**.  Generates meshes for all three benchmark types: Type I (Butterfly), Type II (Rounded Rectangle), and Type III (Eccentric Annulus).
    * `get_cassini_points.m`: Helper function to generate boundary coordinates for the Type I Cassini oval.
    *  `Modulus_zhongkong.m`: Solves the harmonic Dirichlet problem to calculate the **Conformal Modulus** and determine the inner radius $R_{in}$ for Type III apertures (Eq. 6-8).

### Module 2: CSF-QCM Mapping Engine
**Goal:** Solve the mapping $\Phi: \Omega_{phy} \to \Omega_{can}$ using boundary evolution and harmonic relaxation.
*  **Paper Correspondence:** Section 2.3 (Boundary Regularization) & Section 2.4 (Interior Optimization).
* **Key Files:**
    * `CSF_Beltrami_Solver.m`: **Main Engine (Simply Connected)**. Implements Eq. 1 (CSF Evolution) and Eq. 3 (Laplacian Solve) for Type I and II apertures.
    * `csf_zhongkong.m`: **Main Engine (Doubly Connected)**.  Handles the topology-aware mapping for Type III annular apertures.
    *  `Sparse_Laplacian_Matrix.m`: Constructs the cotangent-weight Graph Laplacian matrices ($L_{II}, L_{IB}$) used for mesh relaxation.
    * `CSF_boundary_sequence.m`: Data structure to store and retrieve the sequence of evolved boundary frames $\{\mathcal{C}^{(t)}\}$.

### Module 3: Distortion Analysis - ECDF & Beltrami
**Goal:** Quantify sampling uniformity (ECDF) and verify topological validity (Beltrami coefficient).
*  **Paper Correspondence:** Section 3.1.2 (Area Distortion) & Section 3.2.1 (Quasi-conformal Distortion).
* **Key Files:**
    * **ECDF Analysis:**
        *  `Compare_Boundary_Distortion.m`: Calculates the Normalized Area Ratio $\eta_i$ for every mesh element and plots the **Empirical Cumulative Distribution Function (ECDF)** (Figure 4).
    * **Beltrami Calculation:**
        * `Beltrami_simply.m`: Computes the complex Beltrami coefficient $\mu$ for simply connected meshes.
        * `Beltrami_double.m`: Computes $\mu$ for annular meshes.
        *  `Compare_Beltrami_Heatmap.m`: Visualizes the spatial distribution of $|\mu_{\Phi}|$ to ensure $||\mu||_{\infty} < 1$ (Figure 5).

### Module 4: Numerical Stability - Gram Matrix
**Goal:** Evaluate the recovery of discrete orthogonality of the Zernike basis on the mapped grid.
*  **Paper Correspondence:** Section 3.2.2 (Recovery of Discrete Orthogonality), Figure 6, Table 1.
* **Key Files:**
    * `multi_aperture_gram_analysis.m`: **Master Analysis Script**. Computes the Gram matrix $G = H^T H$ for all aperture types and calculates the Condition Number $\kappa(G)$.
    * `gram_csf_qcm.m`: Function to assemble the Gram matrix using the optimized CSF-QCM grid points.
    * `ricci_flow_gram.m`: Baseline comparison script to compute Gram matrices for the Ricci flow method.

### Module 5: Wavefront Fitting & Residual Analysis
**Goal:** Reconstruct wavefronts, calculate fitting errors (RMS, PV), and test noise robustness.
*  **Paper Correspondence:** Section 3.3 (Wavefront Reconstruction).
* **Key Files:**
    * `verify_csf_qcm_multi.m`: **Master Fitting Script**.
        1.  Loads mapped grids from Module 2.
        2.  Fits wavefronts using Eq.  12: $c = \arg \min \|Hc - W\|_2^2$.
        3.   Calculates Residual RMS and PV errors (Figures 7 & 8).
    * `generate_wavefronts.m`: Generates ground truth wavefronts (e.g., first 36 Zernike modes).
    * `butterfly_csf_fit.m`: Standalone driver for fitting the Type I (Butterfly) aperture.
    * `butterfly_conf_fit.m`: Baseline fitting using standard conformal mapping (demonstrates boundary ringing artifacts).
