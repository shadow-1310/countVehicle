import pandas as pd
import ast
import csv
import torch
import cv2

from sort.sort import *
from countVehicle.utils.common import get_car, read_license_plate, write_csv, interpolate_bounding_boxes, draw_border



class ModelPrediction:
    def __init__(self, vid_path):
        self.cap = cv2.VideoCapture(vid_path)
        # self.cap = vid_path
        self.test_data_path = 'artifacts/test.csv'
        self.interpolated_data_path = 'artifacts/test_interpolated.csv'
        self.output_vid_path = 'artifacts/output.avi'

    def initial_prediction(self):
        cap = self.cap
        results = {}

        mot_tracker = Sort()

        # load models
        coco_model = torch.hub.load("ultralytics/yolov5", 'yolov5s')
        license_plate_detector = torch.hub.load("ultralytics/yolov5", 'custom', 'artifacts/model/best.pt')

        # load video

        vehicles = ['car', 'bus', 'motorcycle', 'bus']

        # read frames
        frame_nmr = -1
        ret = True
        while ret:
            frame_nmr += 1
            ret, frame = cap.read()
            if ret:
                results[frame_nmr] = {}
                # detect vehicles
                detections = coco_model(frame).pandas().xyxy[0]
                detections = detections[detections['name'].isin(vehicles)]
                detections_ = []
                # for detection in detections.boxes.data.tolist():
                for index, row in detections.iterrows():

                    x1 = row['xmin']
                    y1 = row['ymin']
                    x2 = row['xmax']
                    y2 = row['ymax']
                    score = row['confidence']
                    detections_.append([x1, y1, x2, y2, score])

                # track vehicles
                track_ids = mot_tracker.update(np.asarray(detections_))

                # detect license plates
                license_plates = license_plate_detector(frame).pandas().xyxy[0]
                for index, row in license_plates.iterrows():
                    license_plate = row.values.tolist()[:6]
                    x1 = row['xmin']
                    y1 = row['ymin']
                    x2 = row['xmax']
                    y2 = row['ymax']
                    score = row['confidence']

                    # assign license plate to car
                    xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

                    if car_id != -1:

                        # crop license plate
                        license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]

                        # process license plate
                        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                        _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)

                        # read license plate number
                        license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)

                        if license_plate_text is not None:
                            results[frame_nmr][car_id] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                          'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                            'text': license_plate_text,
                                                                            'bbox_score': score,
                                                                            'text_score': license_plate_text_score}}

        # write results
        write_csv(results, self.test_data_path)


    def interpolate_data(self):
        # Load the CSV file
        with open(self.test_data_path, 'r') as file:
            reader = csv.DictReader(file)
            data = list(reader)

        # Interpolate missing data
        interpolated_data = interpolate_bounding_boxes(data)

        # Write updated data to a new CSV file
        header = ['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score', 'license_number', 'license_number_score']
        with open(self.interpolated_data_path, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=header)
            writer.writeheader()
            writer.writerows(interpolated_data)


    def make_visualization(self):
        cap = self.cap
        results = pd.read_csv(self.interpolated_data_path)

        fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')  # Specify the codec
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(self.output_vid_path, fourcc, fps, (width, height))

        license_plate = {}
        for car_id in np.unique(results['car_id']):
            max_ = np.amax(results[results['car_id'] == car_id]['license_number_score'])
            license_plate[car_id] = {'license_crop': None,
                                     'license_plate_number': results[(results['car_id'] == car_id) &
                                                                     (results['license_number_score'] == max_)]['license_number'].iloc[0]}
            cap.set(cv2.CAP_PROP_POS_FRAMES, results[(results['car_id'] == car_id) &
                                                     (results['license_number_score'] == max_)]['frame_nmr'].iloc[0])
            ret, frame = cap.read()

            x1, y1, x2, y2 = ast.literal_eval(results[(results['car_id'] == car_id) &
                                                      (results['license_number_score'] == max_)]['license_plate_bbox'].iloc[0].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))

            license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
            license_crop = cv2.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))

            license_plate[car_id]['license_crop'] = license_crop


        frame_nmr = -1

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # read frames
        ret = True
        while ret:
            ret, frame = cap.read()
            frame_nmr += 1
            if ret:
                df_ = results[results['frame_nmr'] == frame_nmr]
                for row_indx in range(len(df_)):
                    # draw car
                    car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(df_.iloc[row_indx]['car_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
                    draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 25,
                                line_length_x=200, line_length_y=200)

                    # draw license plate
                    x1, y1, x2, y2 = ast.literal_eval(df_.iloc[row_indx]['license_plate_bbox'].replace('[ ', '[').replace('   ', ' ').replace('  ', ' ').replace(' ', ','))
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)

                    # crop license plate
                    license_crop = license_plate[df_.iloc[row_indx]['car_id']]['license_crop']

                    H, W, _ = license_crop.shape

                    try:
                        frame[int(car_y1) - H - 100:int(car_y1) - 100,
                              int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = license_crop

                        frame[int(car_y1) - H - 400:int(car_y1) - H - 100,
                              int((car_x2 + car_x1 - W) / 2):int((car_x2 + car_x1 + W) / 2), :] = (255, 255, 255)

                        (text_width, text_height), _ = cv2.getTextSize(
                            license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'],
                            cv2.FONT_HERSHEY_SIMPLEX,
                            4.3,
                            17)

                        cv2.putText(frame,
                                    license_plate[df_.iloc[row_indx]['car_id']]['license_plate_number'],
                                    (int((car_x2 + car_x1 - text_width) / 2), int(car_y1 - H - 250 + (text_height / 2))),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    4.3,
                                    (0, 0, 0),
                                    17)

                    except:
                        pass

                out.write(frame)
                frame = cv2.resize(frame, (1280, 720))

                # cv2.imshow('frame', frame)
                # cv2.waitKey(0)

        out.release()
        cap.release()
