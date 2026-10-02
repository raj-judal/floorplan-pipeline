"""Compare plane extraction on iPad captures against Faro laser ground truth (ARKitScenes visit 421383).

usage: python research/eval_planes_vs_laser.py LASER_CROP.ply CAPTURE_DIR_42444966 CAPTURE_DIR_42444968
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
B = Path(__file__).resolve().parents[1] / 'benchmarks' / 'arkitscenes_421383'
import numpy as np, open3d as o3d
from floorplan.io import load_capture
from floorplan.geometry.fusion import fuse
from floorplan.geometry.planes import with_normals, horizontal_planes, manhattan_yaw, yaw_matrix, wall_planes
L=o3d.io.read_point_cloud(sys.argv[1])
LP,LN=with_normals(L)
lf,lc=horizontal_planes(LP,LN); yaw=manhattan_yaw(LN,LP,lf); Ry=yaw_matrix(yaw)
cx,cy=np.median(LP[:,0]),np.median(LP[:,1])
print(f'laser: floor tilt {lf.tilt_deg:.2f}deg resid {lf.residual_std*1000:.1f}mm | ceiling tilt {lc.tilt_deg:.2f} resid {lc.residual_std*1000:.1f}mm | height at centre {1000*(lc.z_at(cx,cy)-lf.z_at(cx,cy)):.1f} mm | yaw {np.degrees(yaw):.2f}')
LPy,LNy=LP@Ry.T,LN@Ry.T
# planes are refit in the yaw frame (z unchanged)
lf2,lc2=horizontal_planes(LPy,LNy); lw=[x for x in wall_planes(LPy,LNy,lf2,lc2) if x.structural]
print('laser walls:',[(w.axis,round(w.offset,3),w.n_inliers,round(w.length_supported,2)) for w in lw])
res={}
for name,root,Tf in [('42444966',sys.argv[2],B/'registration_42444966.json'),('42444968',sys.argv[3],B/'registration_42444968.json')]:
    cap=load_capture(root); pc=fuse(cap,stride=2); pc.transform(np.array(json.load(open(Tf))['T_ipad_to_laser']))
    P,N=with_normals(pc)
    f,c=horizontal_planes(P,N); own_yaw=manhattan_yaw(N,P,f)
    Py,Ny=P@Ry.T,N@Ry.T; f2,c2=horizontal_planes(Py,Ny); w=[x for x in wall_planes(Py,Ny,f2,c2) if x.structural]
    H=1000*(c.z_at(cx,cy)-f.z_at(cx,cy))
    print(f'\n{name}: height {H:.1f} mm (err {H-1000*(lc.z_at(cx,cy)-lf.z_at(cx,cy)):+.1f}) | own yaw {np.degrees(own_yaw):.2f} (laser {np.degrees(yaw):.2f})')
    errs=[]
    for g in lw:
        m=[x for x in w if x.axis==g.axis and abs(x.offset-g.offset)<0.08]
        if m:
            x=min(m,key=lambda x:abs(x.offset-g.offset)); errs.append((g,x))
            print(f'  wall axis {g.axis} at {g.offset:+.3f}: ipad {x.offset:+.3f} -> offset error {1000*(x.offset-g.offset):+.1f} mm (support {x.n_inliers})')
        else: print(f'  wall axis {g.axis} at {g.offset:+.3f}: NOT FOUND in ipad')
    res[name]=errs
    # wall-to-wall distances between all matched pairs on same axis, far apart
    for i in range(len(errs)):
        for j in range(i+1,len(errs)):
            gi,xi=errs[i]; gj,xj=errs[j]
            if gi.axis==gj.axis and abs(gi.offset-gj.offset)>1.0:
                print(f'    distance axis {gi.axis} {gi.offset:+.2f}..{gj.offset:+.2f}: laser {abs(gi.offset-gj.offset):.3f} ipad {abs(xi.offset-xj.offset):.3f} err {1000*(abs(xi.offset-xj.offset)-abs(gi.offset-gj.offset)):+.1f} mm')
