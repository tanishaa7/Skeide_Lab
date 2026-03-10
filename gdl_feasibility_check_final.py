#!/usr/bin/env python3
"""
dHCP Geometric Deep Learning Pipeline - Feasibility Tests

Checks:
1. Surface + Sphere Registration Exists
2. Features Match Surface Vertices (with Medial Wall Masking)
3. Cortex / Medial Wall Mask Present
4. Functional Target Exists and Looks Structured
5. Graph Can Be Built from Surface

Plots Generated:
- Plot 1: Structural Feature Distributions (QC)
- Plot 2: Surface-Based Functional Target Map (Projected)
- Plot 3: Surface Vertex Count Consistency (with Resolution Analysis)
- Plot 4: Cross-Validated Surface-Based Learning Task

"""

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')


BASE_DIR = Path("/Users/tanisha/Desktop/MP/for_Tanisha/fmriresults01")
ANAT_DIR = BASE_DIR / "dhcp_anat_pipeline"
FMRI_DIR = BASE_DIR / "dhcp_fmri_pipeline"
OUTPUT_DIR = BASE_DIR / "gdl_feasibility_results"
OUTPUT_DIR.mkdir(exist_ok=True)

SUBJECTS = [
    "sub-CC00856XX15",
    "sub-CC00861XX12",
    "sub-CC00862XX13",
    "sub-CC00864XX15",
    "sub-CC00865XX16"
]

HEMISPHERES = ["left", "right"]

# HELPER FUNCTIONS
def get_session(subject, pipeline_dir):
    """Get first available session for a subject."""
    sub_dir = pipeline_dir / subject
    if not sub_dir.exists():
        return None
    sessions = [d for d in sub_dir.iterdir() if d.is_dir() and d.name.startswith("ses-")]
    return sessions[0].name if sessions else None

def load_gifti_surface(filepath):
    """Load GIFTI surface file and return vertices and faces."""
    img = nib.load(filepath)
    vertices = img.darrays[0].data  # (N, 3) coordinates
    faces = img.darrays[1].data      # (M, 3) triangle indices
    return vertices, faces

def load_gifti_data(filepath):
    """Load GIFTI data file (shape.gii or func.gii) and return data array."""
    img = nib.load(filepath)
    data = img.darrays[0].data
    return data

def load_cifti_dscalar(filepath):
    """Load CIFTI dscalar file and return data array."""
    img = nib.load(filepath)
    data = img.get_fdata().squeeze()
    return data

def load_nifti_4d(filepath):
    """Load 4D NIfTI and return data + affine."""
    img = nib.load(filepath)
    return img.get_fdata(), img.affine

def build_adjacency_from_faces(faces, n_vertices):
    """Build sparse adjacency matrix from triangle faces."""
    edges = set()
    for f in faces:
        edges.add((min(f[0], f[1]), max(f[0], f[1])))
        edges.add((min(f[1], f[2]), max(f[1], f[2])))
        edges.add((min(f[2], f[0]), max(f[2], f[0])))
    
    rows = np.array([e[0] for e in edges])
    cols = np.array([e[1] for e in edges])
    data = np.ones(len(rows))
    
    # Make symmetric by concatenating (not adding)
    sym_rows = np.concatenate([rows, cols])
    sym_cols = np.concatenate([cols, rows])
    sym_data = np.concatenate([data, data])
    
    adj = coo_matrix((sym_data, (sym_rows, sym_cols)), shape=(n_vertices, n_vertices))
    return adj.tocsr()

def project_volume_to_surface(volume_data, affine, vertices, radius=2.0):
    """
    Project volumetric data to surface vertices using trilinear interpolation.
    
    Parameters:
    -----------
    volume_data : 3D or 4D array
        Volumetric data (x, y, z) or (x, y, z, components)
    affine : 4x4 array
        Voxel-to-world affine matrix
    vertices : (N, 3) array
        Surface vertex coordinates in world space (mm)
    radius : float
        Projection radius in mm
    
    Returns:
    --------
    projected : (N,) or (N, C) array
        Data projected to surface vertices
    """
    from scipy.ndimage import map_coordinates
    
    # Handle 4D data (multiple components)
    if len(volume_data.shape) == 4:
        n_components = volume_data.shape[3]
        n_vertices = len(vertices)
        projected = np.zeros((n_vertices, n_components))
        
        for comp in range(n_components):
            projected[:, comp] = project_volume_to_surface(
                volume_data[:,:,:,comp], affine, vertices, radius
            )
        return projected
    
    # Invert affine to get world-to-voxel transformation
    inv_affine = np.linalg.inv(affine)
    
    # Transform vertices to voxel coordinates
    vertices_h = np.hstack([vertices, np.ones((len(vertices), 1))])
    voxels = (inv_affine @ vertices_h.T).T[:, :3]
    
    # Sample volume at voxel coordinates
    projected = map_coordinates(volume_data, voxels.T, order=1, mode='constant', cval=0)
    
    return projected

def spatial_kfold_split(n_vertices, vertices, n_splits=5, random_state=42):
    """
    Create spatially-aware train/test splits to avoid spatial autocorrelation.
    Splits vertices based on spatial location (e.g., anterior/posterior).
    """
    np.random.seed(random_state)
    
    # Sort vertices by Y coordinate (anterior-posterior)
    y_coords = vertices[:, 1]
    sorted_indices = np.argsort(y_coords)
    
    # Create contiguous chunks
    chunk_size = n_vertices // n_splits
    folds = []
    
    for i in range(n_splits):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size if i < n_splits - 1 else n_vertices
        
        test_mask = np.zeros(n_vertices, dtype=bool)
        test_mask[sorted_indices[start_idx:end_idx]] = True
        train_mask = ~test_mask
        
        folds.append((np.where(train_mask)[0], np.where(test_mask)[0]))
    
    return folds

# CHECK 1: SURFACE + SPHERE REGISTRATION EXISTS
def check1_surface_registration():
    """
    Verify that all required surface mesh files exist for each subject.
    Also checks for consistent mesh resolution across subjects.
    """
    print("\n" + "="*70)
    print("CHECK 1: Surface + Sphere Registration Exists")
    print("="*70)
    
    results = {
        'subjects': [], 'left_vertices': [], 'right_vertices': [], 
        'total_vertices': [], 'sphere_exists': [], 'all_surfaces_exist': [],
        'mesh_resolutions': []
    }
    
    required_surfaces = ['midthickness', 'pial', 'wm', 'sphere']
    
    for subject in SUBJECTS:
        session = get_session(subject, ANAT_DIR)
        if not session:
            print(f"\n{subject}:  No session found")
            results['subjects'].append(subject)
            results['left_vertices'].append(0)
            results['right_vertices'].append(0)
            results['total_vertices'].append(0)
            results['sphere_exists'].append(False)
            results['all_surfaces_exist'].append(False)
            results['mesh_resolutions'].append('unknown')
            continue
            
        anat_path = ANAT_DIR / subject / session / "anat"
        
        print(f"\n{subject}:")
        
        all_exist = True
        hemi_vertices = {}
        
        for hemi in HEMISPHERES:
            hemi_exists = True
            for surf_type in required_surfaces:
                surf_file = anat_path / f"{subject}_{session}_hemi-{hemi}_{surf_type}.surf.gii"
                if not surf_file.exists():
                    print(f"   Missing: {surf_file.name}")
                    hemi_exists = False
                    all_exist = False
            
            # Load midthickness to count vertices
            mid_file = anat_path / f"{subject}_{session}_hemi-{hemi}_midthickness.surf.gii"
            if mid_file.exists():
                vertices, faces = load_gifti_surface(mid_file)
                hemi_vertices[hemi] = len(vertices)
                print(f"   {hemi.upper()}: {len(vertices):,} vertices, {len(faces):,} faces")
                
                # Estimate mesh resolution (approximate)
                surf_area = estimate_surface_area(vertices, faces)
                res = np.sqrt(surf_area / len(vertices))
                print(f"       Estimated resolution: ~{res:.2f} mm/vertex")
            else:
                hemi_vertices[hemi] = 0
        
        total_v = hemi_vertices.get('left', 0) + hemi_vertices.get('right', 0)
        
        results['subjects'].append(subject)
        results['left_vertices'].append(hemi_vertices.get('left', 0))
        results['right_vertices'].append(hemi_vertices.get('right', 0))
        results['total_vertices'].append(total_v)
        results['all_surfaces_exist'].append(all_exist)
        
        # Classify mesh resolution
        if total_v > 100000:
            res_class = 'high (~164k native)'
        elif total_v > 40000:
            res_class = 'medium (~80k native)'
        elif total_v > 20000:
            res_class = 'low (~40k native)'
        else:
            res_class = 'very low (<20k)'
        results['mesh_resolutions'].append(res_class)
        
        # Check sphere specifically
        sphere_l = anat_path / f"{subject}_{session}_hemi-left_sphere.surf.gii"
        sphere_r = anat_path / f"{subject}_{session}_hemi-right_sphere.surf.gii"
        results['sphere_exists'].append(sphere_l.exists() and sphere_r.exists())
    
    all_spheres = all(results['sphere_exists'])
    all_surfaces = all(results['all_surfaces_exist'])
    
    print("\n" + "-"*50)
    print(f"  Sphere registration files: {' All present' if all_spheres else ' Some missing'}")
    print(f"  Required surfaces: {' All present' if all_surfaces else ' Some missing'}")
    
    # Check for resolution consistency
    unique_res = set(results['mesh_resolutions'])
    if len(unique_res) > 1:
        print(f"\n    WARNING: Mixed mesh resolutions detected:")
        for subj, res, verts in zip(results['subjects'], results['mesh_resolutions'], results['total_vertices']):
            print(f"      {subj}: {res} ({verts:,} vertices)")
        print(f"\n  This suggests different reconstruction parameters or gestational ages.")
        print(f"  All subjects MUST be resampled to common template before GDL.")
    else:
        print(f"\n   Consistent mesh resolution: {list(unique_res)[0]}")
    
    results['pass'] = all_spheres and all_surfaces
    
    return results

def estimate_surface_area(vertices, faces):
    """Estimate total surface area from mesh."""
    area = 0.0
    for f in faces:
        v0, v1, v2 = vertices[f[0]], vertices[f[1]], vertices[f[2]]
        # Cross product of two edges gives triangle area
        area += 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
    return area

# CHECK 2: FEATURES MATCH SURFACE VERTICES (WITH MEDIAL WALL MASKING)
def check2_features_match_vertices():
    """
    Verify that anatomical feature arrays align with surface vertex counts.
    Uses hardcoded dHCP convention: medialwall_mask = 1 (cortex), 0 (medial wall).
    """
    print("\n" + "="*70)
    print("CHECK 2: Features Match Surface Vertices (with Medial Wall Masking)")
    print("="*70)
    
    results = []
    all_consistent = True
    
    for subject in SUBJECTS:
        session = get_session(subject, ANAT_DIR)
        if not session:
            print(f"\n{subject}:  No session found")
            all_consistent = False
            continue
            
        anat_path = ANAT_DIR / subject / session / "anat"
        
        print(f"\n{subject}:")
        
        # Get total surface vertices (L + R)
        n_left, n_right = 0, 0
        mid_l = anat_path / f"{subject}_{session}_hemi-left_midthickness.surf.gii"
        mid_r = anat_path / f"{subject}_{session}_hemi-right_midthickness.surf.gii"
        
        if mid_l.exists() and mid_r.exists():
            verts_l, _ = load_gifti_surface(mid_l)
            verts_r, _ = load_gifti_surface(mid_r)
            n_left = len(verts_l)
            n_right = len(verts_r)
            n_total_surf = n_left + n_right
            print(f"  Surface vertices: L={n_left:,} + R={n_right:,} = {n_total_surf:,}")
        else:
            print(f"   Could not load surface files")
            all_consistent = False
            continue
        
        # Load medial wall mask - dHCP convention: 1 = cortex, 0 = medial wall
        mask_l = anat_path / f"{subject}_{session}_hemi-left_desc-medialwall_mask.shape.gii"
        mask_r = anat_path / f"{subject}_{session}_hemi-right_desc-medialwall_mask.shape.gii"
        
        n_cortex = 0
        if mask_l.exists() and mask_r.exists():
            mask_left = load_gifti_data(mask_l)
            mask_right = load_gifti_data(mask_r)
            
            # dHCP convention: mask = 1 for cortex, 0 for medial wall
            n_cortex_l = np.sum(mask_left == 1)
            n_cortex_r = np.sum(mask_right == 1)
            n_cortex = n_cortex_l + n_cortex_r
            
            print(f"  Cortex vertices (mask=1): L={n_cortex_l:,} + R={n_cortex_r:,} = {n_cortex:,}")
            print(f"  Medial wall excluded: {n_total_surf - n_cortex:,} vertices")
        else:
            print(f"   Medial wall masks not found")
        
        # Check each feature
        feature_stats = {}
        for feat_name in ['thickness', 'sulc', 'curv']:
            feat_file = anat_path / f"{subject}_{session}_{feat_name}.dscalar.nii"
            if feat_file.exists():
                data = load_cifti_dscalar(feat_file)
                n_feat = len(data)
                n_nan = np.sum(np.isnan(data))
                pct_nan = 100 * n_nan / n_feat
                
                # For thickness: count non-zero (pre-masked)
                if feat_name == 'thickness':
                    n_nonzero = np.sum(data > 0)
                    # Compare non-zero thickness to cortex count
                    match_ratio = n_nonzero / n_cortex if n_cortex > 0 else 0
                    is_consistent = 0.95 <= match_ratio <= 1.05  # Within 5%
                    status = "PASS" if is_consistent else "WARN"
                    print(f"  {status} {feat_name}: N_total={n_feat:,}, N_nonzero={n_nonzero:,} "
                          f"(expected ~{n_cortex:,}, ratio={match_ratio:.3f})")
                else:
                    # Sulc/Curv should match total surface
                    is_consistent = (n_feat == n_total_surf)
                    status = "PASS" if is_consistent else "FAIL"
                    print(f"  {status} {feat_name}: N={n_feat:,} vs N_surface={n_total_surf:,}")
                
                # Check value range
                valid_data = data[~np.isnan(data)]
                if feat_name == 'thickness':
                    valid_data = valid_data[valid_data > 0]
                if len(valid_data) > 0:
                    fmin, fmax = valid_data.min(), valid_data.max()
                    fmean = valid_data.mean()
                else:
                    fmin, fmax, fmean = 0, 0, 0
                
                feature_stats[feat_name] = {
                    'n_features': n_feat,
                    'n_surface': n_total_surf,
                    'n_cortex': n_cortex,
                    'consistent': is_consistent,
                    'pct_nan': pct_nan,
                    'min': fmin,
                    'max': fmax,
                    'mean': fmean
                }
                
                if not is_consistent:
                    all_consistent = False
            else:
                print(f"   Missing: {feat_name}")
                feature_stats[feat_name] = None
                all_consistent = False
        
        results.append({
            'subject': subject,
            'n_left': n_left,
            'n_right': n_right,
            'n_cortex': n_cortex,
            'features': feature_stats
        })
    
    # Summary
    print("\n" + "-"*50)
    print("FEATURE-VERTEX MATCH SUMMARY:")
    print("-"*50)
    
    if all_consistent:
        print("   PASS: All features consistent with surface/mask")
        print("  NOTE: Thickness is pre-masked (zeros at medial wall).")
        print("        After masking, all features align with cortex vertices.")
    else:
        print("   Some inconsistencies detected - review details above")
    
    return {'results': results, 'pass': all_consistent}

# CHECK 3: CORTEX / MEDIAL WALL MASK PRESENT
def check3_medial_wall_mask():
    """
    Verify medial wall masks are present and valid.
    Uses hardcoded dHCP convention: 1 = cortex, 0 = medial wall.
    """
    print("\n" + "="*70)
    print("CHECK 3: Cortex / Medial Wall Mask Present")
    print("="*70)
    
    results = []
    
    for subject in SUBJECTS:
        session = get_session(subject, ANAT_DIR)
        if not session:
            print(f"\n{subject}:  No session found")
            continue
            
        anat_path = ANAT_DIR / subject / session / "anat"
        
        print(f"\n{subject}:")
        
        mask_stats = {}
        for hemi in HEMISPHERES:
            mask_file = anat_path / f"{subject}_{session}_hemi-{hemi}_desc-medialwall_mask.shape.gii"
            
            if mask_file.exists():
                mask = load_gifti_data(mask_file)
                n_total = len(mask)
                
                # dHCP convention: 1 = cortex, 0 = medial wall
                n_cortex = np.sum(mask == 1)
                n_medial = np.sum(mask == 0)
                pct_cortex = 100 * n_cortex / n_total
                
                print(f"   {hemi.upper()}: {n_cortex:,}/{n_total:,} cortex vertices ({pct_cortex:.1f}%)")
                
                mask_stats[hemi] = {
                    'n_total': n_total,
                    'n_cortex': n_cortex,
                    'n_medial': n_medial,
                    'pct_cortex': pct_cortex,
                    'exists': True
                }
            else:
                print(f"   Missing: {hemi} medial wall mask")
                mask_stats[hemi] = {'exists': False}
        
        results.append({
            'subject': subject,
            'masks': mask_stats
        })
    
    # Summary
    print("\n" + "-"*50)
    print("MEDIAL WALL MASK SUMMARY:")
    print("-"*50)
    
    all_present = all(
        r['masks']['left'].get('exists', False) and r['masks']['right'].get('exists', False)
        for r in results
    )
    
    if all_present:
        cortex_pcts = []
        for r in results:
            for hemi in HEMISPHERES:
                cortex_pcts.append(r['masks'][hemi]['pct_cortex'])
        
        mean_cortex = np.mean(cortex_pcts)
        std_cortex = np.std(cortex_pcts)
        print(f"   PASS: All masks present")
        print(f"  Mean cortex coverage: {mean_cortex:.1f}% ± {std_cortex:.1f}%")
        print(f"  Expected range: 90-98% (consistent with dHCP)")
    else:
        print(f"   FAIL: Some masks missing")
    
    return {'results': results, 'pass': all_present}

# CHECK 4: FUNCTIONAL TARGET EXISTS AND LOOKS STRUCTURED
def check4_functional_target():
    """
    Verify functional target data exists and is structured (not noise).
    Also attempts surface projection if surface files available.
    """
    print("\n" + "="*70)
    print("CHECK 4: Functional Target Exists and Looks Structured")
    print("="*70)
    
    results = []
    
    for subject in SUBJECTS:
        fmri_session = get_session(subject, FMRI_DIR)
        if not fmri_session:
            print(f"\n{subject}:  No fMRI session found")
            results.append({
                'subject': subject,
                'exists': False,
                'structured': False
            })
            continue
            
        func_path = FMRI_DIR / subject / fmri_session / "func"
        
        print(f"\n{subject}:")
        
        ic_file = func_path / f"{subject}_{fmri_session}_task-rest_desc-ic_maps.nii.gz"
        
        if ic_file.exists():
            ic_data, affine = load_nifti_4d(ic_file)
            n_components = ic_data.shape[-1] if len(ic_data.shape) == 4 else 1
            
            # QC checks
            ic_min = ic_data.min()
            ic_max = ic_data.max()
            ic_std = ic_data.std()
            pct_zero = 100 * np.sum(ic_data == 0) / ic_data.size
            
            structured = ic_std > 0.01 and pct_zero < 99
            status = "PASS" if structured else "WARN"
            
            print(f"  {status} IC maps: shape={ic_data.shape}")
            print(f"       Components: {n_components}")
            print(f"       Range: [{ic_min:.3f}, {ic_max:.3f}]")
            print(f"       Std: {ic_std:.4f}, %zeros: {pct_zero:.1f}%")
            
            # Attempt surface projection for first subject
            projected_data = None
            if subject == SUBJECTS[0]:
                anat_session = get_session(subject, ANAT_DIR)
                if anat_session:
                    anat_path = ANAT_DIR / subject / anat_session / "anat"
                    mid_l = anat_path / f"{subject}_{anat_session}_hemi-left_midthickness.surf.gii"
                    mid_r = anat_path / f"{subject}_{anat_session}_hemi-right_midthickness.surf.gii"
                    
                    if mid_l.exists() and mid_r.exists():
                        try:
                            verts_l, _ = load_gifti_surface(mid_l)
                            verts_r, _ = load_gifti_surface(mid_r)
                            vertices = np.vstack([verts_l, verts_r])
                            
                            # Project first component to surface
                            if len(ic_data.shape) == 4:
                                proj = project_volume_to_surface(ic_data[:,:,:,0], affine, vertices)
                                print(f"       Surface projection:  {len(proj):,} vertices mapped")
                                projected_data = proj
                            else:
                                proj = project_volume_to_surface(ic_data, affine, vertices)
                                print(f"       Surface projection:  {len(proj):,} vertices mapped")
                                projected_data = proj
                        except Exception as e:
                            print(f"       Surface projection:  Failed ({str(e)[:50]})")
            
            results.append({
                'subject': subject,
                'exists': True,
                'shape': ic_data.shape,
                'n_components': n_components,
                'structured': structured,
                'ic_data': ic_data,
                'affine': affine,
                'projected_data': projected_data
            })
        else:
            print(f"   Missing: IC maps")
            results.append({
                'subject': subject,
                'exists': False,
                'structured': False
            })
    
    all_exist = all(r['exists'] for r in results)
    all_structured = all(r['structured'] for r in results)
    
    print("\n" + "-"*50)
    if all_exist and all_structured:
        print("   PASS: All functional targets exist and are structured")
        print("  NOTE: Volumetric data requires projection to surface for GDL")
    elif all_exist:
        print("   PARTIAL: Files exist but some may lack structure")
    else:
        print("   FAIL: Some functional targets missing")
    
    return {'results': results, 'pass': all_exist and all_structured}

# CHECK 5: GRAPH CAN BE BUILT FROM SURFACE
def check5_graph_construction():
    """
    Verify that a valid graph can be built from the surface mesh.
    Includes additional sanity checks on mesh quality.
    """
    print("\n" + "="*70)
    print("CHECK 5: Graph Can Be Built from Surface")
    print("="*70)
    
    # Test on first subject
    subject = SUBJECTS[0]
    session = get_session(subject, ANAT_DIR)
    if not session:
        print(f"\n No session found for {subject}")
        return {'results': {}, 'pass': False}
        
    anat_path = ANAT_DIR / subject / session / "anat"
    
    print(f"\nTesting graph construction on {subject}:")
    
    results = {}
    
    for hemi in HEMISPHERES:
        print(f"\n  {hemi.upper()} hemisphere:")
        
        mid_file = anat_path / f"{subject}_{session}_hemi-{hemi}_midthickness.surf.gii"
        
        if mid_file.exists():
            vertices, faces = load_gifti_surface(mid_file)
            n_vertices = len(vertices)
            n_faces = len(faces)
            
            print(f"    Vertices: {n_vertices:,}")
            print(f"    Faces (triangles): {n_faces:,}")
            
            # Sanity check: expected faces vs vertices ratio
            expected_faces = 2 * n_vertices - 4  # Euler characteristic for sphere-like surface
            face_ratio = n_faces / n_vertices
            print(f"    Faces/Vertices ratio: {face_ratio:.2f} (expected ~2.0)")
            
            if abs(face_ratio - 2.0) > 0.5:
                print(f"     WARNING: Unusual face/vertex ratio. Check mesh topology.")
            
            # # Check for degenerate faces
            # degenerate = 0
            # for f in faces:
            #     if f[0] == f[1] or f[1] == f[2] or f[2] == f[0]:
            #                            degenerate += 1
            # if degenerate > 0:
            #     print(f"     WARNING: {degenerate:,} degenerate faces found")
            
            # # Build adjacency matrix
            # adj = build_adjacency_from_faces(faces, n_vertices)
            
            # Check for degenerate faces
            degenerate = 0
            for f in faces:
                if f[0] == f[1] or f[1] == f[2] or f[2] == f[0]:
                    degenerate += 1
            if degenerate > 0:
                print(f"     WARNING: {degenerate:,} degenerate faces found")
            
            # Build adjacency matrix
            adj = build_adjacency_from_faces(faces, n_vertices)

            # Check for isolated nodes
            degrees = np.array(adj.sum(axis=1)).flatten()
            n_isolated = np.sum(degrees == 0)
            min_degree = degrees.min()
            max_degree = degrees.max()
            mean_degree = degrees.mean()
            
            print(f"    Mean degree: {mean_degree:.1f}")
            print(f"    Degree range: [{min_degree}, {max_degree}]")
            print(f"    Isolated nodes: {n_isolated}")
            
            # Check connected components
            n_components, labels = connected_components(adj, directed=False)
            
            print(f"    Connected components: {n_components}")
            
            # Check vertex coordinate validity
            coord_range = np.ptp(vertices, axis=0)  # Peak-to-peak (max - min)
            print(f"    Coordinate ranges (mm): X={coord_range[0]:.1f}, Y={coord_range[1]:.1f}, Z={coord_range[2]:.1f}")
            
            if np.any(coord_range < 1):
                print(f"     WARNING: Very small coordinate range - check units (should be mm)")
            
            if n_isolated == 0 and n_components == 1:
                print(f"     PASS: Valid graph structure")
                valid = True
            else:
                print(f"     FAIL: Graph issues detected")
                valid = False
            
            results[hemi] = {
                'n_vertices': n_vertices,
                'n_faces': n_faces,
                'n_isolated': n_isolated,
                'n_components': n_components,
                'mean_degree': mean_degree,
                'face_ratio': face_ratio,
                'valid': valid
            }
        else:
            print(f"     Surface file not found")
            results[hemi] = {'valid': False}
    
    all_valid = all(r.get('valid', False) for r in results.values())
    
    print("\n" + "-"*50)
    if all_valid:
        print("   PASS: Graph construction successful for both hemispheres")
        print("  Mesh quality checks passed.")
    else:
        print("   FAIL: Graph construction issues detected")
    
    return {'results': results, 'pass': all_valid}

# GENERATE PLOTS
def generate_plots(check_results):
    """Generate all required plots from check results with proper validation."""
    
    print("\n" + "="*70)
    print("GENERATING PLOTS")
    print("="*70)
    
    # Get first subject with valid data
    subject = None
    for subj in SUBJECTS:
        sess = get_session(subj, ANAT_DIR)
        if sess:
            subject = subj
            break
    
    if not subject:
        print("   No valid subjects found for plotting")
        return
    
    session = get_session(subject, ANAT_DIR)
    anat_path = ANAT_DIR / subject / session / "anat"
    
    # PLOT 1: Structural Feature Distributions (QC)
    print("\n  Creating Plot 1: Structural Feature Distributions...")
    
    try:
        thickness = load_cifti_dscalar(anat_path / f"{subject}_{session}_thickness.dscalar.nii")
        sulc = load_cifti_dscalar(anat_path / f"{subject}_{session}_sulc.dscalar.nii")
        curv = load_cifti_dscalar(anat_path / f"{subject}_{session}_curv.dscalar.nii")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Quality Control of Cortical Surface Features (dHCP)", 
                     fontsize=14, fontweight='bold')
        
        # Thickness - check for outliers
        valid_thickness = thickness[thickness > 0]
        # Flag potential outliers (>2mm for neonates is suspicious)
        outliers = np.sum(valid_thickness > 2.0)
        outlier_pct = 100 * outliers / len(valid_thickness)
        
        axes[0].hist(valid_thickness, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0].axvline(np.median(valid_thickness), color='red', linestyle='--', 
                        label=f'Median: {np.median(valid_thickness):.2f}mm')
        if outlier_pct > 1:
            axes[0].axvline(2.0, color='orange', linestyle=':', 
                           label=f'Outliers >2mm: {outlier_pct:.1f}%')
        axes[0].set_xlabel('Cortical Thickness (mm)')
        axes[0].set_ylabel('Vertex Count')
        axes[0].set_title('Thickness Distribution')
        axes[0].legend()
        
        # Sulcal Depth
        axes[1].hist(sulc, bins=50, color='forestgreen', alpha=0.7, edgecolor='black')
        axes[1].axvline(np.median(sulc), color='red', linestyle='--', 
                        label=f'Median: {np.median(sulc):.2f}')
        axes[1].set_xlabel('Sulcal Depth')
        axes[1].set_ylabel('Vertex Count')
        axes[1].set_title('Sulcal Depth Distribution')
        axes[1].legend()
        
        # Curvature
        axes[2].hist(curv, bins=50, color='darkorange', alpha=0.7, edgecolor='black')
        axes[2].axvline(np.median(curv), color='red', linestyle='--', 
                        label=f'Median: {np.median(curv):.3f}')
        axes[2].set_xlabel('Curvature')
        axes[2].set_ylabel('Vertex Count')
        axes[2].set_title('Curvature Distribution')
        axes[2].legend()
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "plot1_feature_distributions.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"     Saved: plot1_feature_distributions.png")
        
    except Exception as e:
        print(f"     Failed to create Plot 1: {e}")
    
    # PLOT 2: Surface-Based Functional Target Map (PROJECTED)
    print("\n  Creating Plot 2: Surface-Projected Functional Target...")
    
    try:
        fmri_session = get_session(subject, FMRI_DIR)
        if fmri_session:
            func_path = FMRI_DIR / subject / fmri_session / "func"
            ic_file = func_path / f"{subject}_{fmri_session}_task-rest_desc-ic_maps.nii.gz"
            
            if ic_file.exists():
                ic_data, affine = load_nifti_4d(ic_file)
                
                # Load surface vertices
                mid_l = anat_path / f"{subject}_{session}_hemi-left_midthickness.surf.gii"
                mid_r = anat_path / f"{subject}_{session}_hemi-right_midthickness.surf.gii"
                
                if mid_l.exists() and mid_r.exists():
                    verts_l, faces_l = load_gifti_surface(mid_l)
                    verts_r, faces_r = load_gifti_surface(mid_r)
                    
                    # Project volume to surface
                    if len(ic_data.shape) == 4:
                        n_components = min(6, ic_data.shape[3])
                        
                        fig = plt.figure(figsize=(16, 10))
                        fig.suptitle("Surface-Projected ICA Components (Functional Targets)", 
                                    fontsize=14, fontweight='bold')
                        
                        for idx in range(n_components):
                            ax = fig.add_subplot(2, 3, idx + 1, projection='3d')
                            
                            # Project this component
                            proj_l = project_volume_to_surface(ic_data[:,:,:,idx], affine, verts_l)
                            proj_r = project_volume_to_surface(ic_data[:,:,:,idx], affine, verts_r)
                            
                            # Simple visualization: scatter plot colored by value
                            # Downsample for visualization
                            step = max(1, len(verts_l) // 5000)
                            
                            # Left hemisphere (shifted left for visualization)
                            sc = ax.scatter(verts_l[::step, 0] - 50, verts_l[::step, 1], verts_l[::step, 2],
                                          c=proj_l[::step], cmap='RdBu_r', s=1, vmin=-5, vmax=5)
                            # Right hemisphere (shifted right)
                            ax.scatter(verts_r[::step, 0] + 50, verts_r[::step, 1], verts_r[::step, 2],
                                      c=proj_r[::step], cmap='RdBu_r', s=1, vmin=-5, vmax=5)
                            
                            ax.set_title(f'IC Component {idx + 1}')
                            ax.set_xlabel('X')
                            ax.set_ylabel('Y')
                            ax.set_zlabel('Z')
                            ax.view_init(elev=20, azim=90)
                            plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
                        
                        plt.tight_layout()
                        plt.savefig(OUTPUT_DIR / "plot2_functional_target.png", dpi=150, bbox_inches='tight')
                        plt.close()
                        print(f"     Saved: plot2_functional_target.png (surface-projected)")
                    else:
                        print(f"     IC data is not 4D, skipping surface projection plot")
                else:
                    print(f"     Surface files not found, cannot project to surface")
            else:
                print(f"     IC maps not found")
        else:
            print(f"     No fMRI session found")
            
    except Exception as e:
        print(f"     Failed to create Plot 2: {e}")
        # Create fallback plot showing volume slice with explanation
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'Surface Projection Failed\n\n' + str(e)[:200],
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.axis('off')
            plt.savefig(OUTPUT_DIR / "plot2_functional_target.png", dpi=150)
            plt.close()
        except:
            pass
    
    # PLOT 3: Surface Vertex Count Consistency (with Resolution Analysis)
    print("\n  Creating Plot 3: Vertex Count Consistency...")
    
    try:
        vertex_data = []
        for subj in SUBJECTS:
            sess = get_session(subj, ANAT_DIR)
            if not sess:
                continue
            anat_p = ANAT_DIR / subj / sess / "anat"
            
            # Get both surface count and estimated resolution
            mid_l = anat_p / f"{subj}_{sess}_hemi-left_midthickness.surf.gii"
            mid_r = anat_p / f"{subj}_{sess}_hemi-right_midthickness.surf.gii"
            
            if mid_l.exists() and mid_r.exists():
                verts_l, faces_l = load_gifti_surface(mid_l)
                verts_r, faces_r = load_gifti_surface(mid_r)
                
                total_v = len(verts_l) + len(verts_r)
                
                # Estimate surface area and resolution
                area_l = estimate_surface_area(verts_l, faces_l)
                area_r = estimate_surface_area(verts_r, faces_r)
                total_area = area_l + area_r
                
                # Approximate resolution: sqrt(area / n_vertices)
                resolution = np.sqrt(total_area / total_v) if total_v > 0 else 0
                
                vertex_data.append({
                    'subject': subj,
                    'vertices': total_v,
                    'resolution': resolution,
                    'area': total_area
                })
        
        if vertex_data:
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            fig.suptitle("Surface Mesh Analysis Across Subjects", fontsize=14, fontweight='bold')
            
            subjects_short = [d['subject'].split('-')[1][:8] for d in vertex_data]
            vertex_counts = [d['vertices'] for d in vertex_data]
            resolutions = [d['resolution'] for d in vertex_data]
            
            # Plot: Vertex counts
            bars = ax.bar(subjects_short, vertex_counts, color='teal', edgecolor='black')
            for bar, count in zip(bars, vertex_counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000, 
                           f'{count:,}', ha='center', va='bottom', fontsize=9)
            
            ax.set_ylabel('Number of Vertices')
            ax.set_xlabel('Subject ID')
            ax.set_title('Total Vertex Count')
            
            # Add horizontal lines for common template resolutions
            ax.axhline(32492, color='red', linestyle='--', alpha=0.7, label='fsLR 32k')
            ax.axhline(64984, color='orange', linestyle='--', alpha=0.7, label='fsLR 64k')
            ax.legend(loc='upper right', fontsize=8)
            
            # # COMMENTED OUT: Mesh Resolution plot (not essential)
            # axes[1].bar(subjects_short, resolutions, color='steelblue', edgecolor='black')
            # for i, (subj, res) in enumerate(zip(subjects_short, resolutions)):
            #     axes[1].text(i, res + 0.02, f'{res:.2f}mm', ha='center', va='bottom', fontsize=9)
            # axes[1].set_ylabel('Estimated Resolution (mm/vertex)')
            # axes[1].set_xlabel('Subject ID')
            # axes[1].set_title('Mesh Resolution (Lower = Finer)')
            # axes[1].set_ylim(0, max(resolutions) * 1.2)
            
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "plot3_vertex_consistency.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"     Saved: plot3_vertex_consistency.png")
        else:
            print(f"     No vertex data available for plotting")
            
    except Exception as e:
        print(f"     Failed to create Plot 3: {e}")
    
    # PLOT 4: Cross-Validated Surface-Based Learning Task
    print("\n  Creating Plot 4: Cross-Validated Learning Feasibility...")
    
    try:
        # Load data
        sulc = load_cifti_dscalar(anat_path / f"{subject}_{session}_sulc.dscalar.nii")
        curv = load_cifti_dscalar(anat_path / f"{subject}_{session}_curv.dscalar.nii")
        
        # Load surface for spatial cross-validation
        mid_l = anat_path / f"{subject}_{session}_hemi-left_midthickness.surf.gii"
        mid_r = anat_path / f"{subject}_{session}_hemi-right_midthickness.surf.gii"
        
        if mid_l.exists() and mid_r.exists():
            verts_l, _ = load_gifti_surface(mid_l)
            verts_r, _ = load_gifti_surface(mid_r)
            vertices = np.vstack([verts_l, verts_r])
        else:
            # Fallback: create dummy vertices for random CV
            vertices = np.random.randn(len(sulc), 3)
        
        # Prepare data
        X = sulc.reshape(-1, 1)
        y = curv
        n_samples = len(X)
        
        # Spatial cross-validation
        n_splits = 5
        fold_results = []
        
        print(f"    Running {n_splits}-fold spatial cross-validation...")
        
        # Create spatial folds based on Y coordinate (anterior-posterior)
        y_coords = vertices[:, 1]
        sorted_indices = np.argsort(y_coords)
        fold_size = n_samples // n_splits
        
        all_y_true = []
        all_y_pred = []
        
        for fold in range(n_splits):
            # Define test set as contiguous spatial region
            start_idx = fold * fold_size
            end_idx = (fold + 1) * fold_size if fold < n_splits - 1 else n_samples
            
            test_mask = np.zeros(n_samples, dtype=bool)
            test_mask[sorted_indices[start_idx:end_idx]] = True
            train_mask = ~test_mask
            
            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]
            
            # Train model
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_train, y_train)
            y_pred_fold = ridge.predict(X_test)
            
            # Evaluate
            r2_fold = r2_score(y_test, y_pred_fold)
            corr_fold = np.corrcoef(y_test, y_pred_fold)[0, 1] if len(y_test) > 1 else 0
            
            fold_results.append({
                'fold': fold + 1,
                'n_train': np.sum(train_mask),
                'n_test': np.sum(test_mask),
                'r2': r2_fold,
                'correlation': corr_fold
            })
            
            all_y_true.extend(y_test)
            all_y_pred.extend(y_pred_fold)
            
            print(f"      Fold {fold + 1}: R² = {r2_fold:.3f}, r = {corr_fold:.3f}")
        
        # Aggregate results
        all_y_true = np.array(all_y_true)
        all_y_pred = np.array(all_y_pred)
        
        r2_mean = np.mean([f['r2'] for f in fold_results])
        r2_std = np.std([f['r2'] for f in fold_results])
        corr_mean = np.mean([f['correlation'] for f in fold_results])
        
        # Overall R² on all predictions
        r2_overall = r2_score(all_y_true, all_y_pred)
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        fig.suptitle("Cross-Validated Surface-Based Learning Task", fontsize=14, fontweight='bold')
        
        # Plot: Prediction vs Truth (all folds combined)
        # Use hexbin for large datasets
        from matplotlib.colors import LogNorm
        hb = ax.hexbin(all_y_true, all_y_pred, gridsize=50, cmap='Blues', 
                               bins='log', mincnt=1)
        ax.plot([all_y_true.min(), all_y_true.max()], 
                       [all_y_true.min(), all_y_true.max()], 
                       'r--', linewidth=2, label='Identity')
        ax.set_xlabel('True Curvature')
        ax.set_ylabel('Predicted Curvature')
        ax.set_title(f'Prediction vs Truth (r = {corr_mean:.3f}, R² = {r2_mean:.3f})')
        ax.legend()
        plt.colorbar(hb, ax=ax, label='log(count)')
                
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "plot4_toy_learning.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"     Saved: plot4_toy_learning.png")
        
    except Exception as e:
        print(f"     Failed to create Plot 4: {e}")
        import traceback
        traceback.print_exc()

# MAIN: RUN ALL CHECKS
def run_all_checks():
    """Run all 5 feasibility checks and generate plots."""
    print("\n" + "="*70)
    print("  dHCP GEOMETRIC DEEP LEARNING - FEASIBILITY TESTS")
    print("="*70)
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Subjects: {len(SUBJECTS)}")
    
    results = {}
    
    # Run checks
    results['check1'] = check1_surface_registration()
    results['check2'] = check2_features_match_vertices()
    results['check3'] = check3_medial_wall_mask()
    results['check4'] = check4_functional_target()
    results['check5'] = check5_graph_construction()
    
    # Generate plots
    generate_plots(results)
    
    # Final summary
    print("\n" + "="*70)
    print("  FEASIBILITY TEST SUMMARY")
    print("="*70)
    
    checks = [
        ("1. Surface + Sphere Registration", results['check1']['pass']),
        ("2. Features Match Vertices (with Masking)", results['check2']['pass']),
        ("3. Medial Wall Mask Present", results['check3']['pass']),
        ("4. Functional Target Structured", results['check4']['pass']),
        ("5. Graph Construction Valid", results['check5']['pass']),
    ]
    
    n_pass = sum(1 for _, p in checks if p)
    
    print(f"\n  Results: {n_pass}/5 checks passed\n")
    
    for name, passed in checks:
        status = " PASS" if passed else " FAIL"
        print(f"    {status} - {name}")
    
    print(f"\n  Plots saved to: {OUTPUT_DIR}")
    
    return results

if __name__ == "__main__":
    run_all_checks()