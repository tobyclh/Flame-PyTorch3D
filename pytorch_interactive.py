from pytorch3d.utils import ico_sphere
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance
import torch


import os
import torch
import matplotlib.pyplot as plt
import numpy as np

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesVertex
)

import os
from smplx import FLAME
import streamlit as st
device = 'cuda'

if '_is_init' not in st.session_state or not st.session_state['_is_init']:
    model_folder = '/home/toby/Documents/FCS/Core/sample_data/FLAME2020/'
    gender = 'generic'
    ext = 'pkl'
    st.session_state['shape'] = torch.zeros([1, 300], dtype=torch.float32)
    st.session_state['exp'] = torch.zeros([1, 100], dtype=torch.float32)
    model = FLAME(model_folder, gender=gender, ext=ext, num_betas=300, num_expression_coeffs=100)
    st.session_state['flame_model'] = model
    
    output = model(betas=st.session_state['shape'], expression=st.session_state['exp'], return_verts=True)
    vertices = output.vertices.detach().to(device).squeeze()
    faces = torch.tensor(model.faces.astype(np.int32), dtype=torch.long, device=device)
    verts_rgb = torch.ones_like(vertices)[None] # (1, V, 3)
    textures = TexturesVertex(verts_rgb)
    st.session_state['textures'] = textures
    mesh = Meshes(verts=[vertices], faces=[faces], textures=textures)
    st.session_state['mesh'] = mesh
    R, T = look_at_view_transform(0.6, 0, 0) 
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=40)

    raster_settings = RasterizationSettings(
        image_size=512, 
        blur_radius=0.0, 
        faces_per_pixel=1, 
    )

    lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device, 
            cameras=cameras,
            lights=lights
        )
    )
    st.set_page_config(layout="wide")
    st.session_state['renderer'] = renderer
    st.session_state['_is_init'] = True
def change_color():
    color = st.session_state['color'].lstrip('#')
    color = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
    for i in range(3):
        st.session_state['textures']._verts_features_padded[..., i] = color[i]/255
    # textures = TexturesVertex(verts_rgb)
    # st.session_state['mesh'].textures = textures

st.sidebar.color_picker('Vert color', value='#439fe0', on_change=change_color,  key='color')

col1, col2, col3 = st.columns([1, 1, 2])


with col1.expander('Shape sliders'):
    start_index, end_index = col1.slider('Shape axis', min_value=0, max_value=299, value=[0, 10])
    # start_index = st.number_input('start from', min_value=0, max_value=299-n_slider, value=10)
    for idx in range(start_index, end_index+1):
        st.session_state['shape'][0, idx] = col1.slider(f'shape_{idx}', min_value=-2.0, max_value=2.0, value=0.0)

with col2.expander('Exp sliders'):
    start_index, end_index = col2.slider('Exp axis', min_value=0, max_value=99, value=[0, 10])
    # start_index = st.number_input('start from', min_value=0, max_value=299-n_slider, value=10)
    for idx in range(start_index, end_index+1):
        st.session_state['exp'][0, idx] = col2.slider(f'exp_{idx}', min_value=-2.0, max_value=2.0, value=0.0)


with torch.no_grad():
    output = st.session_state['flame_model'](betas=st.session_state['shape'], expression=st.session_state['exp'], return_verts=True)
    st.session_state['mesh']._verts_padded = output.vertices.detach().to(device)
    images = st.session_state['renderer'](st.session_state['mesh'])
fig, ax = plt.subplots()
col3.image(images[0, ..., :3].cpu().numpy(), use_column_width=True)
# ax.imshow()
# # plt.axis("off")
# col3.pyplot(fig)