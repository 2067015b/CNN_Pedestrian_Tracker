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
                for coordinate in line[1:-1]:
                    current_det.append(int(coordinate))
                current_det.append(line[-1])
                dets = self.gt.get(current_frame,[])
                dets.append(current_det)
                self.gt[current_frame] = dets

        self.video_path = video_path
        self.frame = frame
        self.accumulator = mm.MOTAccumulator(auto_id=True)

    def log(self, frame_no, detections):
        gt_detections = self.gt.get(frame_no,[])
        detection_boxes = [detection.get_detections(frame_no) for detection in detections]
        detection_boxes = [list([detection[1],detection[0],detection[3]-detection[1],detection[2]-detection[0]]) for detection in detection_boxes]
        # if not detections or not gt_detections:
        #     return

        print("METRIC {}: Detections: {}, gts: {}".format(frame_no,detection_boxes,gt_detections))
        gt_detection_boxes = [det[:-1] for det in gt_detections]
        distances = mm.distances.iou_matrix(gt_detection_boxes,detection_boxes,max_iou=0.5)
        print(distances)

        # alphabet = 'abcdefghijklmnopqrstuvxyz'
        gt_detections = [det[-1] for det in gt_detections]
        print(gt_detections)
        detections = [det.id for det in detections]

        self.accumulator.update(gt_detections,detections,distances)

    def print_stats(self):
        mh = mm.metrics.create()
        summary = mh.compute_many(
            [self.accumulator, self.accumulator.events.loc[0:1]],
            metrics=mm.metrics.motchallenge_metrics,
            names=['full', 'part'])

        strsummary = mm.io.render_summary(
            summary,
            formatters=mh.formatters,
            namemap=mm.io.motchallenge_metric_names
        )
        print(strsummary)

        print(mh.compute(self.accumulator, metrics=['num_frames', 'mota', 'motp'], name='acc'))




