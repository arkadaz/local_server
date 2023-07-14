import tensorrt as trt
import os


class TensorRTConversion:
    def __init__(
        self, path_onnx, path_tensorrt, max_workspace_size=1 << 30, half_precision=False
    ) -> None:
        self.TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        self.path_onnx = path_onnx
        self.path_tensorrt = path_tensorrt
        self.max_workspace_size = max_workspace_size
        self.half_precision = half_precision

    def convert(self):
        builder = trt.Builder(self.TRT_LOGGER)
        config = builder.create_builder_config()
        config.max_workspace_size = self.max_workspace_size
        explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

        network = builder.create_network(explicit_batch)
        parser = trt.OnnxParser(network, self.TRT_LOGGER)

        with open(self.path_onnx, "rb") as model_onnx:
            if not parser.parse(model_onnx.read()):
                print("ERROR: FAIL TO PARSE ONNX MODEL")
                for error in parser.errors:
                    print(error)
                return None

        profile = builder.create_optimization_profile()
        profile.set_shape(
            "input_name",
            min=(1, 3, 256, 256),
            opt=(1, 3, 256, 256),
            max=(1, 3, 256, 256),
        )
        config.add_optimization_profile(profile)
        print("SUCCESSFULLY TEOSORRT ENGINE TO MAX BATCH")

        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)

        engine = builder.build_engine(network, config)
        with open(self.path_tensorrt, "wb") as f_engine:
            f_engine.write(engine.serialize())

        print("SUCCESSFULLY ALL STEP")


path = os.getcwd()
convert = TensorRTConversion(
    "{}/model/GLUE.onnx".format(path), "{}/model/GLUE.trt".format(path)
).convert()
