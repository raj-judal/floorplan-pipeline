import numpy as np, glob, os, sys, open3d as o3d
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
root, Tf = sys.argv[1], sys.argv[2]
T=np.load(Tf)
L=o3d.io.read_point_cloud('laser_room_crop.ply'); L.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.05,max_nn=30))
LP,LN=np.asarray(L.points),np.asarray(L.normals); tree=cKDTree(LP)
traj=np.loadtxt(f'{root}/lowres_wide.traj')
Rwc=np.transpose(R.from_rotvec(traj[:,1:4]).as_matrix(),(0,2,1)); C=-np.einsum('nij,nj->ni',Rwc,traj[:,4:7])
intr={float(os.path.basename(f).split('_')[1][:-7]):f for f in glob.glob(f'{root}/lowres_wide_intrinsics/*.pincam')}; ik=np.array(sorted(intr))
rows=[]
for f in sorted(glob.glob(f'{root}/lowres_depth/*.png'))[::4]:
    ts=float(os.path.basename(f).split('_')[1][:-4]); j=np.argmin(abs(traj[:,0]-ts))
    if abs(traj[j,0]-ts)>0.02: continue
    w,h,fx,fy,cx,cy=np.loadtxt(intr[ik[np.argmin(abs(ik-ts))]])
    d=np.array(Image.open(f)).astype(np.float32)/1000; c=np.array(Image.open(f.replace('lowres_depth','confidence')))
    v,u=np.nonzero((c>=1)&(d>0.2)&(d<5)); z=d[v,u]
    P=np.stack([(u+0.5-cx)/fx*z,(v+0.5-cy)/fy*z,z],1)
    Rt=T[:3,:3]@Rwc[j]; cam=T[:3,:3]@C[j]+T[:3,3]
    W=P@Rt.T+cam; rng=np.linalg.norm(P,axis=1); ray=(W-cam)/rng[:,None]
    dist,idx=tree.query(W,distance_upper_bound=0.06); ok=np.isfinite(dist)
    n=LN[idx[ok]]; cosi=np.abs(np.sum(n*ray[ok],1))
    sd=np.sum((W[ok]-LP[idx[ok]])*n,1)*np.sign(-np.sum(n*ray[ok],1))   # >0 means point is in front of the laser surface (toward camera)
    e_ray=sd/np.maximum(cosi,0.3)
    rows.append(np.stack([rng[ok],e_ray,c[v,u][ok],cosi],1))
A=np.concatenate(rows); A=A[A[:,3]>0.6]
print(f'{root}: {len(A):,} points matched to laser surface (steep views only)')
for conf in [2,1]:
    B=A[A[:,2]==conf]; out=[]
    for a,b in [(0.3,0.75),(0.75,1.25),(1.25,1.75),(1.75,2.5),(2.5,3.5),(3.5,5)]:
        s=B[(B[:,0]>=a)&(B[:,0]<b)]
        if len(s)>2000: out.append(f'{a}-{b}m: {1000*np.median(s[:,1]):+.1f}mm')
    print(f'  conf={conf}: '+' | '.join(out))
