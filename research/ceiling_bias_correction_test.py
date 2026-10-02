import numpy as np, glob, os, sys
from PIL import Image
from scipy.spatial.transform import Rotation as R
FL,CE=192.4954,194.8251; lo,hi=np.array([-0.31,-2.82])+0.5,np.array([7.2,0.89])-0.5
# bias curve calibrated on capture 42444966 only (conf=2 medians, metres, range bin centres)
BR=np.array([0.5,1.0,1.5,2.1,3.0]); BB=np.array([0.0198,0.0109,0.0140,0.0109,0.0054])
def run(root,Tf,correct):
    T=np.load(Tf); traj=np.loadtxt(f'{root}/lowres_wide.traj')
    Rwc=np.transpose(R.from_rotvec(traj[:,1:4]).as_matrix(),(0,2,1)); C=-np.einsum('nij,nj->ni',Rwc,traj[:,4:7])
    intr={float(os.path.basename(f).split('_')[1][:-7]):f for f in glob.glob(f'{root}/lowres_wide_intrinsics/*.pincam')}; ik=np.array(sorted(intr))
    zs={0:[],1:[]}
    for f in sorted(glob.glob(f'{root}/lowres_depth/*.png'))[::2]:
        ts=float(os.path.basename(f).split('_')[1][:-4]); j=np.argmin(abs(traj[:,0]-ts))
        if abs(traj[j,0]-ts)>0.02: continue
        w,h,fx,fy,cx,cy=np.loadtxt(intr[ik[np.argmin(abs(ik-ts))]])
        d=np.array(Image.open(f)).astype(np.float32)/1000; c=np.array(Image.open(f.replace('lowres_depth','confidence')))
        v,u=np.nonzero((c==2)&(d>0.2)&(d<5)); z=d[v,u]
        P=np.stack([(u+0.5-cx)/fx*z,(v+0.5-cy)/fy*z,z],1)
        if correct:
            r=np.linalg.norm(P,axis=1); P=P*(1+np.interp(r,BR,BB)/r)[:,None]
        W=(P@Rwc[j].T+C[j])@T[:3,:3].T+T[:3,3]
        inner=np.all((W[:,:2]>lo)&(W[:,:2]<hi),1)
        for k,z0 in [(0,FL),(1,CE)]: zs[k].append(W[inner&(np.abs(W[:,2]-z0)<0.06),2])
    f,c=np.median(np.concatenate(zs[0])),np.median(np.concatenate(zs[1]))
    return 1000*(c-f)
for root,Tf,cap in [('ark','T_ipad_to_laser.npy','42444966 (calibration)'),('ark2','T_ipad68_to_laser.npy','42444968 (held out)')]:
    a,b=run(root,Tf,False),run(root,Tf,True)
    print(f'{cap}: raw {a:.1f} mm (err {a-2329.6:+.1f}) -> corrected {b:.1f} mm (err {b-2329.6:+.1f})')
