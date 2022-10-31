from ast import parse
from pytorch3d.structures import Meshes

import torch
import numpy as np

from pytorch3d.structures import Meshes
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

from smplx import FLAME
import streamlit as st
from argparse import ArgumentParser


if '_is_init' not in st.session_state or not st.session_state['_is_init']:
    st.set_page_config(layout="wide")
    st.session_state['_is_init'] = True
    
    parser = ArgumentParser('PyTorch3D interactive app for FLAME model')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--img_size', default=512, type=int, help='rendering resolution')
    parser.add_argument('--fov', default=30, type=int, help='field of view for camera')
    parser.add_argument('--gender', default='generic', type=str, choices=['generic', 'female', 'male'], help='gender of flame model')

    args = parser.parse_args()
    device = args.device
    st.session_state['device'] = device

    
    model_folder = 'data/FLAME2020/'
    gender = args.gender
    ext = 'pkl'
    for i in range(300):
        st.session_state[f'shape_{i}'] = 0
    for i in range(100):
        st.session_state[f'exp_{i}'] = 0
    for i in range(3):
        st.session_state[f'jaw_{i}'] = 0
        st.session_state[f'neck_{i}'] = 0
    model = FLAME(model_folder, gender=gender, ext=ext, num_betas=300, num_expression_coeffs=100)
    st.session_state['flame_model'] = model
    np_shape = np.array([st.session_state[f'shape_{i}'] for i in range(300)], dtype=np.float32) 
    np_exps = np.array([st.session_state[f'exp_{i}'] for i in range(100)], dtype=np.float32) 
    np_jaw = np.array([st.session_state[f'jaw_{i}'] for i in range(3)], dtype=np.float32) 
    np_neck = np.array([st.session_state[f'neck_{i}'] for i in range(3)], dtype=np.float32) 
    betas = torch.tensor(np_shape, device='cpu', dtype=torch.float)[None]
    exps = torch.tensor(np_exps, device='cpu', dtype=torch.float)[None]
    jaw = torch.tensor(np_jaw, device='cpu', dtype=torch.float)[None]
    neck = torch.tensor(np_neck, device='cpu', dtype=torch.float)[None]
    output = model(betas=betas, expression=exps, neck_pose=neck, jaw_pose=jaw, return_verts=True)
    vertices = output.vertices.detach().to(device).squeeze()
    faces = torch.tensor(model.faces.astype(np.int32), dtype=torch.long, device=device)
    verts_rgb = torch.ones_like(vertices)[None] # (1, V, 3)
    verts_rgb[..., 0] = 69/255
    verts_rgb[..., 1] = 145/255
    verts_rgb[..., 2] = 197/255
    textures = TexturesVertex(verts_rgb)
    st.session_state['textures'] = textures
    mesh = Meshes(verts=[vertices], faces=[faces], textures=textures)
    st.session_state['mesh'] = mesh
    
    R, T = look_at_view_transform(0.6, 0, 0) 
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=args.fov)

    raster_settings = RasterizationSettings(
        image_size=args.img_size, 
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
    
    st.session_state['renderer'] = renderer
    
st.title('Flame model viewer')

def change_color():
    color = st.session_state['color'].lstrip('#')
    color = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
    for i in range(3):
        st.session_state['textures']._verts_features_padded[..., i] = color[i]/255

st.sidebar.color_picker('Vert color', value='#439fe0', on_change=change_color,  key='color')

col1, col2, col3 = st.columns([1, 1, 2])

with col1.expander('Neck'):
    for idx in range(3):
        st.slider(f'neck_{idx}', min_value=-0.5, max_value=0.5, value=0.0, key=f'neck_{idx}')

with col2.expander('Jaw'):
    for idx in range(3):
        st.slider(f'jaw_{idx}', min_value=-0.5, max_value=0.5, value=0.0, key=f'jaw_{idx}')

with col1.expander('Shape sliders'):
    start_index, end_index = st.slider('Shape axis', min_value=0, max_value=299, value=[0, 3])
    for idx in range(start_index, end_index+1):
        st.slider(f'shape_{idx}', min_value=-2.0, max_value=2.0, value=0.0, key=f'shape_{idx}')

with col2.expander('Exp sliders'):
    start_index, end_index = st.slider('Exp axis', min_value=0, max_value=99, value=[0, 3])
    for idx in range(start_index, end_index+1):
        st.slider(f'exp_{idx}', min_value=-2.0, max_value=2.0, value=0.0, key=f'exp_{idx}')

with torch.no_grad():
    np_shape = np.array([st.session_state[f'shape_{i}'] for i in range(300)], dtype=np.float32) 
    np_exps = np.array([st.session_state[f'exp_{i}'] for i in range(100)], dtype=np.float32) 
    np_jaw = np.array([st.session_state[f'jaw_{i}'] for i in range(3)], dtype=np.float32) 
    np_neck = np.array([st.session_state[f'neck_{i}'] for i in range(3)], dtype=np.float32) 
    betas = torch.tensor(np_shape, device='cpu', dtype=torch.float)[None]
    exps = torch.tensor(np_exps, device='cpu', dtype=torch.float)[None]
    jaw = torch.tensor(np_jaw, device='cpu', dtype=torch.float)[None]
    neck = torch.tensor(np_neck, device='cpu', dtype=torch.float)[None]
    output = st.session_state['flame_model'](betas=betas, expression=exps, neck_pose=neck, jaw_pose=jaw, return_verts=True)
    st.session_state['mesh']._verts_padded = output.vertices.detach().to(st.session_state['device'])
    images = st.session_state['renderer'](st.session_state['mesh'])

use_column_width = not st.sidebar.checkbox('original_size', value=False)
col3.image(images[0, ..., :3].cpu().numpy(), use_column_width=use_column_width)
# ax.imshow()
# # plt.axis("off")
# col3.pyplot(fig)