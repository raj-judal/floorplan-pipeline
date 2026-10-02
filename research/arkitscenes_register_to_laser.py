import numpy as np, open3d as o3d, copy, time
src=o3d.io.read_point_cloud('ipad_42444966.ply'); tgt=o3d.io.read_point_cloud('laser_room_crop.ply')
v=0.08
def prep(pc):
    d=pc.voxel_down_sample(v); d.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=v*2,max_nn=30))
    return d,o3d.pipelines.registration.compute_fpfh_feature(d,o3d.geometry.KDTreeSearchParamHybrid(radius=v*5,max_nn=100))
s,sf=prep(src); t,tf=prep(tgt); print('down',len(s.points),len(t.points),flush=True)
best=None
for seed in range(4):
    o3d.utility.random.seed(seed); t0=time.time()
    r=o3d.pipelines.registration.registration_ransac_based_on_feature_matching(s,t,sf,tf,True,v*1.5,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(v*1.5)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(200000,0.999))
    print('seed',seed,'fitness',round(r.fitness,3),'rmse',round(r.inlier_rmse,4),'t',round(time.time()-t0,1),flush=True)
    if best is None or r.fitness>best.fitness: best=r
T=best.transformation
src_d=src.voxel_down_sample(0.02); tgt.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.05,max_nn=30))
for thr in [0.15,0.06,0.025]:
    icp=o3d.pipelines.registration.registration_icp(src_d,tgt,thr,T,o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=50))
    T=icp.transformation; print(f'icp thr {thr}: fitness {icp.fitness:.3f} rmse {icp.inlier_rmse*1000:.1f} mm',flush=True)
np.save('T_ipad_to_laser.npy',T)
Rm=T[:3,:3]; print('det',round(np.linalg.det(Rm),6),'tilt deg',round(np.degrees(np.arccos(np.clip(Rm[2,2],-1,1))),3))
d=np.asarray(copy.deepcopy(src_d).transform(T).compute_point_cloud_distance(tgt))
print('residual median %.1f mm, 75th %.1f mm, 90th %.1f mm'%(1000*np.median(d),1000*np.percentile(d,75),1000*np.percentile(d,90)),flush=True)
print('DONE')
