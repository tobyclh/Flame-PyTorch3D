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
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)

# add path for demo utils functions 
import sys
import os
# sys.path.append(os.path.abspath(''))


# Use an ico_sphere mesh and load a mesh from an .obj e.g. model.obj
sphere_mesh = ico_sphere(level=3)
verts, faces, _ = load_obj("cube.obj")
print(faces.verts_idx.shape)

from smplx import FLAME

model_folder = '/home/toby/Documents/FCS/Core/sample_data/FLAME2020/'
gender = 'generic'
ext = 'pkl'
model = FLAME(model_folder, gender=gender, ext=ext, num_betas=10, num_expression_coeffs=10)
betas = torch.zeros([1, 10], dtype=torch.float32)
expression = torch.zeros([1, 10], dtype=torch.float32)

output = model(betas=betas, expression=expression, 
                return_verts=True)
vertices = output.vertices.detach().cpu().squeeze()
faces = torch.tensor(model.faces.astype(np.int32), dtype=torch.long) #, device='cuda')
verts_rgb = torch.ones_like(vertices)[None] # (1, V, 3)
textures = TexturesVertex(verts_rgb)

mesh = Meshes(verts=[vertices], faces=[faces], textures=textures)
device = 'cpu'
# joints = output.joints.detach().cpu().numpy().squeeze()

print('Vertices shape =', vertices.shape)
print('Joints shape =', faces.shape)

# Initialize a camera.
# With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction. 
# So we move the camera by 180 in the azimuth direction so it is facing the front of the cow. 
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

images = renderer(mesh)
# plt.figure(figsize=(10, 10))
plt.imshow(images[0, ..., :3].cpu().numpy())
plt.axis("off")
plt.show()