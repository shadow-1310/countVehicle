from countVehicle.components.model_prediction import ModelPrediction

class ModelPredictionPipeline:
    def __init__(self, vid_path):
        self.vid_path = vid_path

    def main(self):
        m = ModelPrediction(self.vid_path)
        m.initial_prediction()
        m.interpolate_data()
        m.make_visualization()
        m.find_count()


# if __name__ == '__main__':
#     vid_path = 'artifacts/video/sample.mp4'
#     pipe = ModelPredictionPipeline(vid_path)
#     pipe.main()
