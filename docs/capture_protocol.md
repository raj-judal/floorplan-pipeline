# Capture protocol (one page)
Pick ONE tier. Follow every step exactly. Questions about a step mean the step is wrong: tell us.
## LiDAR tier (iPhone Pro or iPad Pro with LiDAR)
1. Install **Stray Scanner** (free, App Store). Turn on all room lights. Open every interior door fully.
2. Stand in the entrance doorway. Hold the phone upright at chest height. Tap record.
3. Walk slowly, about half a normal walking pace, through **every room**, stepping fully inside each one.
4. In each room, slowly sweep the camera across each wall **from floor to ceiling**, then across the ceiling once.
5. Do not point the camera at mirrors for more than a moment. Keep fingers off the camera lenses.
6. Finish back in the entrance doorway where you started, facing the way you first faced. Stop recording. Expect 2-5 minutes.
7. Hand-off: connect the phone to the computer with a cable; open Finder (Mac) or iTunes (Windows), select the phone, open the Files tab, and drag the newest folder under Stray Scanner (a name like `c7d28f72c6`) to the computer.
8. Run: `python -m floorplan run PATH\TO\FOLDER --tier lidar`
## Photo tier (any iPhone 15 or newer)
1. Turn on all room lights. In the Camera app, choose **Photo** and tap **0.5x**. Hold the phone **sideways (landscape)**.
2. For each room, stand with your **back touching the middle of a wall**, phone at chest height, tilted **slightly down** so the floor shows at the bottom of the screen. Take one photo straight across the room.
3. Repeat from the middle of **each wall** (4 photos), then take 2-4 more from the same spots. 6-8 photos per room.
4. An L-shaped room counts as two rooms: photograph each rectangular part separately.
5. Hand-off: make one folder per room on the computer (any names, e.g. `kitchen`, `bedroom1`) and copy that room's photos into it, unedited. Put all room folders inside one capture folder.
6. Run: `python -m floorplan run PATH\TO\CAPTURE_FOLDER --tier photo`
## Video tier (any iPhone 15 or newer)
1. Same set-up as photos: lights on, Camera app, **Video**, **0.5x**, phone **sideways**.
2. Record **one clip per room**: start with your back to the middle of a wall, film straight across for 2 seconds, then walk to the middle of the next wall and repeat for all four walls. Turn slowly. 20-40 seconds per room.
3. Hand-off: copy the clips into one folder, one clip per room, unedited (any names).
4. Run: `python -m floorplan run PATH\TO\CAPTURE_FOLDER --tier video`
## What each step is for (evidence in docs/lab_notebook.md)
- Enter every room; sweep walls to the ceiling: rooms not entered, or with walls seen only below ~2 m, were missed or merged in the sample scans (floor-only scan had 4x fewer wall points above 1.5 m).
- End where you started: lets drift correction close the loop; the scans that ended away from the start drifted more.
- 0.5x, landscape, back to the wall, slightly down: in rendered tests, level 1x photos never showed the floor (no scale); 0.5x tilted ~10 deg down showed the floor in 4 of 4 views.
- 6-8 photos per room: scale accuracy per room is about 12% with 3 photos, 7.7% with 5, 5.6% with 8 (90% of rooms).
- Chest height: real-world scale for photos and video assumes the camera is ~1.40 m above the floor.
