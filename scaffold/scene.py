import json
import random
import shutil
from pathlib import Path

from scene.dataset_readers import sceneLoadTypeCallbacks
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON


class Scene:
    def __init__(self, args, model, shuffle=False, initialize_model=True, save_metadata=True):
        self.model_path = Path(args.model_path)

        if (Path(args.source_path) / "sparse").exists():
            info = sceneLoadTypeCallbacks["Colmap"](
                args.source_path, args.images, args.eval, args.lod)
        elif (Path(args.source_path) / "transforms_train.json").exists():
            info = sceneLoadTypeCallbacks["Blender"](
                args.source_path, args.white_background, args.eval)
        else:
            raise ValueError(f"Unrecognized scene at {args.source_path}")

        if save_metadata:
            self.model_path.mkdir(parents=True, exist_ok=True)
            input_ply = self.model_path / "input.ply"
            if initialize_model or not input_ply.exists():
                shutil.copyfile(info.ply_path, input_ply)
            cameras = info.test_cameras + info.train_cameras
            with (self.model_path / "cameras.json").open("w", encoding="utf-8") as file:
                json.dump([camera_to_JSON(i, camera)
                           for i, camera in enumerate(cameras)], file)

        if shuffle:
            random.shuffle(info.train_cameras)
            random.shuffle(info.test_cameras)

        print("Loading training cameras")
        self.train_cameras = cameraList_from_camInfos(
            info.train_cameras, 1.0, args)
        print("Loading test cameras")
        self.test_cameras = cameraList_from_camInfos(
            info.test_cameras, 1.0, args)
        self.cameras_extent = info.nerf_normalization["radius"]
        if initialize_model:
            model.set_appearance(len(info.train_cameras))
            model.create_from_pcd(info.point_cloud, self.cameras_extent)

    def save(self, iteration, model):
        path = self.model_path / "point_cloud" / f"iteration_{iteration}"
        model.save_ply(path / "point_cloud.ply")
        model.save_mlp_checkpoints(path)

    def getTrainCameras(self):
        return self.train_cameras

    def getTestCameras(self):
        return self.test_cameras
