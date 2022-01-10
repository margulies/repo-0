
### VARIABLES TO SET BEFORE RUNNING
# directory containing subdirectories named fter subject IDs that contain the timeseries and surface files
root_dir = 
# directory where all intermediate files and the final output will be saved
output_dir = 
# list of IDs of subjects to include in the analyses
subj_id = np.array([])

#-----------------------------------------------------------------------------------

import nibabel as nib
import numpy as np
from scipy import stats
import xml.etree.ElementTree as xml
import pandas as pd
from nilearn.plotting import plot_surf_stat_map
from matplotlib import pyplot as plt


# https://github.com/AthenaEPI/logpar/blob/master/logpar/utils/cifti_utils.py
def extract_matrixIndicesMap(cifti_header, direction):
    ''' Retrieves the xml of a Matrix Indices Map from a cifti file.
       Parameters
       ----------
       cifti_header: cifti header
           Header of the cifti file
       direction: string
           ROW or COLUMN
       Returns
       -------
       xml_entitie
       Returns an xml (Elemtree) object
    '''
    dim = 0 if direction == 'ROW' else 1

    cxml = xml.fromstring(cifti_header.extensions[0].get_content().to_xml())

    query = ".//MatrixIndicesMap[@AppliesToMatrixDimension='{}']".format(dim)
    matrix_indices_map = cxml.find(query)

    return matrix_indices_map


def extract_brainmodel(cifti_header, structure, direction):
    '''
       Retrieves the xml of a brain model structure from a cifti file.
       Parameters
       ----------
       cifti_header: cifti header
           Header of the cifti file
       structure: string
           Name of structure
       direction: string
           ROW or COLUMN
       Returns
       -------
       xml_entitie
       Returns an xml (Elemtree) object
    '''
    matrix_indices = extract_matrixIndicesMap(cifti_header, direction)

    if structure == 'ALL':
        query = "./BrainModel"
        brain_model = matrix_indices.findall(query)
    else:
        query = "./BrainModel[@BrainStructure='{}']".format(structure)
        brain_model = matrix_indices.findall(query)
    return brain_model


def extract_zone_convergence(sub, hemi, brain_structure):
    # load surf
    surf_raw = nib.load(f'{root_dir}{sub}/T1w/fsaverage_LR32k/{sub}.{hemi}.midthickness_MSMAll.32k_fs_LR.surf.gii')
    surf = []
    surf.append(surf_raw.darrays[0].data)
    surf.append(surf_raw.darrays[1].data)
    vertices, triangles = surf#[surf_raw.darrays[i].data for i in range(2)]

    # load labels
    labels = nib.load(f'{root_dir}{sub}/{sub}.zone_prim.32k_fs_LR.dlabel.nii')
    zones = labels.get_fdata().squeeze()

    brain_model = extract_brainmodel(labels.header, brain_structure, 'COLUMN')
    offset = int(brain_model[0].attrib['IndexOffset'])
    cortex = np.array(brain_model[0].find('VertexIndices').text.split(), dtype=int)

    z = np.zeros(vertices.shape[0])
    z[cortex] = zones[offset:offset+len(cortex)]
    # next step takes advantage of prod of 1,2,3 == 6.
    coords = np.argwhere(np.prod(z[triangles], axis=1) == 6.)

    # take more posterior node:
    trig_of_interest = np.argmin([vertices[triangles[coords[0]]][0][:,1].mean(), vertices[triangles[coords[1]]][0][:,1].mean()])
    nodes_of_interest = triangles[coords[trig_of_interest]][0]
    
    return nodes_of_interest


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


### Find the scalar value associated to a set of vetices and a their percentile in a specified structure
def get_scalar_pctile(cifti_scalar, vertices, brain_structure, scalar_row=0):
    
    # extract scalar values of all vertices, and the features of the brain structure from cifti
    all_scalars = np.array(cifti_scalar.get_fdata()[scalar_row])
    brain_model = [x for x in cifti_scalar.header.get_index_map(1).brain_models if x.brain_structure==brain_structure]
    offset = brain_model[0].index_offset
    count = brain_model[0].index_count
    vertex_indices = np.array(brain_model[0].vertex_indices)
    idx = np.array([i for i,x in enumerate(vertex_indices) if x in vertices])
    
    # get scalars and relative percentile
    vertex_scalars = all_scalars[offset+idx]
    scalar_pctiles = [stats.percentileofscore(all_scalars[offset:offset+count],scalar) for scalar in vertex_scalars]
    del cifti_scalar
    
    return vertex_scalars, scalar_pctiles

#-----------------------------------------------------------------------------------

### Obtain the gradient values corresponding to the convergence nodes and relative percentile
print("Extracting principal gradient and percentile of convergence nodes")

# pre-assign the output dataframe
gradientile_df = pd.DataFrame(columns=['ID_vtx','ID_grad','hemisphere',
                                         'vertex1','vertex2','vertex3',
                                         'value1','value2','value3',
                                         'mean','percentile'])

for subj_grad in subj_id:
    # load a subject's gradient1
    hemisphere = {'L':'CIFTI_STRUCTURE_CORTEX_LEFT', 'R':'CIFTI_STRUCTURE_CORTEX_RIGHT'}
    grads = nib.load(f'{output_dir}{subj_grad}.REST1_gcca.dscalar.nii')
    
    # get gradient and percentile of convergence nodes of all subjs from the current grad
    for subj_vtx in subj_id:        
        for hemi in hemisphere.keys():           
            brain_structure = hemisphere[hemi]
            vtx_of_interest = extract_zone_convergence(subj_vtx, hemi, brain_structure)
            vtx_of_interest = np.sort(vtx_of_interest.tolist(),axis=0)
            vtx_grads, vtx_pctile = get_scalar_pctile(grads, vtx_of_interest, brain_structure)
            # update output dataframe
            gradientile_df = gradientile_df.append({'ID_vtx':subj_vtx,'ID_grad':subj_grad,'hemisphere':hemi,
                                                    'vertex1':vtx_of_interest[0],'vertex2':vtx_of_interest[1],'vertex3':vtx_of_interest[2],
                                                    'value1':vtx_grads[0],'value2':vtx_grads[1],'value3':vtx_grads[2],
                                                    'mean':np.mean(vtx_grads),'percentile':np.mean(vtx_pctile)}, ignore_index=True)
    del grads
# save output        
gradientile_df.to_csv(f'{output_dir}gradientiles.csv',index=False)
print("Done!")

#-----------------------------------------------------------------------------------


