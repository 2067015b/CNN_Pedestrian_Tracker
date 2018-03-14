import numpy as np
import motmetrics as mm


class Metric:

    def __init__(self, gt, video_path=None, frame=0):
        self.gt = {}
        with open(gt, 'r') as gt_file:
            for line in gt_file:
                line = line.split()
                current_frame = int(float(line[0]))
                current_det = []
                for coordinate in line[1:]:
                    current_det.append(int(coordinate))
                dets = self.gt.get(current_frame,[])
                dets.append(current_det)
                self.gt[current_frame] = dets

        self.video_path = video_path
        self.frame = frame
        self.accumulator = mm.MOTAccumulator(auto_id=True)

    def log(self, frame_no, detections):
        gt_detections = self.gt.get(frame_no,[])
        if not detections or not gt_detections:
            return


        distances = mm.distances.iou_matrix(gt_detections,detections,max_iou=0.5)

        alphabet = 'abcdefghijklmnopqrstuvxyz'
        gt_detections = [alphabet[i] for i,_ in enumerate(gt_detections)]
        detections = [i+1 for i,_ in enumerate(detections)]

        self.accumulator.update(gt_detections,detections,distances)

    def print_stats(self):
        mh = mm.metrics.create()
        print(mh.compute(self.accumulator, metrics=['num_frames', 'mota', 'motp', 'tp', 'tn', 'fn', 'fp'], name='acc'))




