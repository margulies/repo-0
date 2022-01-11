### VARIABLES TO SET BEFORE RUNNING
# directory containing subdirectories named fter subject IDs that contain the timeseries and surface files
root_dir = "/home/fralberti/Data/HCP/"
# directory where all intermediate files and the final output will be saved
output_dir = "/home/fralberti/Data/Gradientile/"
# list of IDs of subjects to include in the analyses
subj_id = np.array([])

#-----------------------------------------------------------------------------------

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
import surfdist as sd
from surfdist import viz, load, utils, analysis
import hcp_utils as hcp
import pandas as pd
from scipy import stats
from nilearn.plotting import plot_surf_stat_map
from itertools import combinations

### Store gradients in cifti2 dscalar.nii
### Store gradients in cifti2 dscalar.nii
def mk_grad1_dscalar(grads, template_cifti, output_dir):
    # grads: array with dimensions gradients X vertices
    # template_cifti: any cifti2 file with a BrainModelAxis (I am using one of the dtseries.nii)
    # output_dir: path to output directory
    data = np.zeros([grads.shape[0],template_cifti.shape[1]])
    data[0:,0:grads.shape[1]] = grads

    map_labels = [f'Measure {i+1}' for i in range(grads.shape[0])]
    ax0 = nib.cifti2.cifti2_axes.ScalarAxis(map_labels)
    ax1 = nib.cifti2.cifti2_axes.from_index_mapping(template_cifti.header.get_index_map(1))
    nifti_hdr = template_cifti.nifti_header
    del template_cifti
    
    new_img = nib.Cifti2Image(data, header=[ax0, ax1],nifti_header=nifti_hdr)
    new_img.update_headers()

    new_img.to_filename(output_dir)

def get_parcel(label,path_to_dlabel,brain_structure):
    ### Get the vertex index of nodes included in a specified parcel
    # label: label of the parcel of which you need the nodes
    # path_to_dlabel: path to a cifti2.dlabel.nii file
    # brain_structure: cifti2 brain structure
    
    cifti = nib.load(path_to_dlabel)
    brain_model = [x for x in cifti.header.get_index_map(1).brain_models if x.brain_structure==brain_structure][0]
    offset = brain_model.index_offset
    count = brain_model.index_count
    vertices = np.array(brain_model.vertex_indices[0:])

    label_map = cifti.get_fdata().squeeze()[offset:offset+count]

    label_lst = pd.DataFrame(cifti.header.get_axis(0).get_element(0)[1].values(),columns=('lab','col')).reset_index()
    label_tmp = label_lst[label_lst['lab'].isin(label)]['index'].values
    
    del cifti
    return [lab in label_tmp for lab in label_map], vertices[[lab in label_tmp for lab in label_map]]

#-----------------------------------------------------------------------------------

### Find the transmodal peak of the principal gradient in the lateral parietal area

peaks = pd.DataFrame(columns=['ID','L_vtx','R_vtx','L_grad','R_grad'])

for i,subj in enumerate(subj_id):
    roi = []
    grad = nib.load(f'{output_dir}{subj}.REST1_gcca.dscalar.nii')
    for hemi in ['L','R']:
        # define a mask to limit peak search to the lateral parietal and occipital cortex
        label = [f'{hemi}_postcentral',f'{hemi}_supramarginal',f'{hemi}_inferiorparietal',f'{hemi}_superiorparietal',f'{hemi}_lateraloccipital']
        path_to_dlabel = f'{root_dir}{subj}/MNINonLinear/fsaverage_LR32k/{subj}.aparc.32k_fs_LR.dlabel.nii'
        brain_structure = ['CIFTI_STRUCTURE_CORTEX_LEFT' if hemi=='L' else 'CIFTI_STRUCTURE_CORTEX_RIGHT'][0]
        mask, vtx = get_parcel(label,path_to_dlabel,brain_structure)
        roi.extend(mask)
        # find vertex with the highest gradient value (all gradients must follow the uni-to-transmodal direction)
        bm = [x for x in grad.header.get_index_map(1).brain_models if x.brain_structure==brain_structure][0]
        grad_hemi = grad.get_fdata()[0,bm.index_offset:bm.index_offset+bm.index_count][mask]
        vertices = np.array(bm.vertex_indices)[mask]
        peak_tmp = vertices[np.where(grad_hemi==grad_hemi.max())[0][0]]
        peaks.loc[i,['ID',hemi+'_vtx',hemi+'_grad']] = [subj,peak_tmp,grad_hemi.max()]
#     mk_grad1_dscalar(np.array(roi,ndmin=2), grad, f'/home/fralberti/Data/Gradientile/{subj}.peak_mask.dscalar.nii')
peaks.to_csv(f'{output_dir}gradient1_peaks.csv',index=False)

del grad

#-----------------------------------------------------------------------------------

grad_peaks = pd.read_csv(f'{output_dir}gradient1_peaks.csv')
gradientiles = pd.read_csv(f'{output_dir}grad1entiles.csv')
int_vtx = gradientiles[(gradientiles.ID_vtx==gradientiles.ID_grad)][['ID_vtx','hemisphere','vertex1','vertex2','vertex3']]
del gradientiles

peak_dist_df = pd.DataFrame(columns=['ID_peak','ID_int','hemisphere','distance'])
for subj_peak in subj_id:
    for hemi in ['L','R']:
        surf = nib.load(f'{root_dir}{subj_peak}/T1w/fsaverage_LR32k/{subj_peak}.{hemi}.midthickness_MSMAll.32k_fs_LR.surf.gii').darrays
        coord = [surf[0].data,surf[1].data]
        nodes = np.unique(surf[1].data)
        for subj_int in subj_id:
            src_vtx = int_vtx[(int_vtx.ID_vtx==int(subj_int)) & (int_vtx.hemisphere==hemi)][vertex1','vertex2','vertex3']]
            peak_tmp = grad_peaks.loc[grad_peaks.ID==int(subj_peak),f'{hemi}_vtx']
            dist_tmp = sd.analysis.dist_calc(coord, nodes, src_vtx)
            peak_dist_df = peak_dist_df.append({'ID_peak':subj_peak,'ID_int':subj_int,'hemisphere':hemi,
                                                'distance':dist_tmp[peak_tmp][0]}, ignore_index=True)
            del dist_tmp
        del surf, coord, nodes
        
peak_dist_df.sort_values(['ID_peak','ID_int'],'Axis'==1).to_csv(f'{output_dir}peak_dist.csv',index=False)

#-----------------------------------------------------------------------------------