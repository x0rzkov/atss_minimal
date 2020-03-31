#!/usr/bin/env python 
# -*- coding:utf-8 -*-

# The following per-class thresholds are computed by maximizing
# per-class f-measure in their precision-recall curve.
# Please see compute_thresholds_for_classes() in coco_eval.py for details.
thre_per_classes = [0.4923645853996277, 0.4928510785102844, 0.5040897727012634,
                    0.4912887513637543, 0.5016880631446838, 0.5278812646865845,
                    0.5351834893226624, 0.5003424882888794, 0.4955945909023285,
                    0.43564629554748535, 0.6089804172515869, 0.666087806224823,
                    0.5932040214538574, 0.48406165838241577, 0.4062422513961792,
                    0.5571075081825256, 0.5671307444572449, 0.5268378257751465,
                    0.5112953186035156, 0.4647842049598694, 0.5324517488479614,
                    0.5795850157737732, 0.5152440071105957, 0.5280804634094238,
                    0.4791383445262909, 0.5261335372924805, 0.4906163215637207,
                    0.523737907409668, 0.47027698159217834, 0.5103300213813782,
                    0.4645252823829651, 0.5384289026260376, 0.47796186804771423,
                    0.4403403103351593, 0.5101461410522461, 0.5535093545913696,
                    0.48472103476524353, 0.5006796717643738, 0.5485560894012451,
                    0.4863888621330261, 0.5061569809913635, 0.5235867500305176,
                    0.4745445251464844, 0.4652363359928131, 0.4162440598011017,
                    0.5252017974853516, 0.42710989713668823, 0.4550687372684479,
                    0.4943239390850067, 0.4810051918029785, 0.47629663348197937,
                    0.46629616618156433, 0.4662836790084839, 0.4854755401611328,
                    0.4156557023525238, 0.4763634502887726, 0.4724511504173279,
                    0.4915047585964203, 0.5006274580955505, 0.5124194622039795,
                    0.47004589438438416, 0.5374764204025269, 0.5876904129981995,
                    0.49395060539245605, 0.5102297067642212, 0.46571290493011475,
                    0.5164387822151184, 0.540651798248291, 0.5323763489723206,
                    0.5048757195472717, 0.5302401781082153, 0.48333442211151123,
                    0.5109739303588867, 0.4077408015727997, 0.5764586925506592,
                    0.5109297037124634, 0.4685552418231964, 0.5148998498916626,
                    0.4224434792995453, 0.4998510777950287]

class_names = ['__background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
               'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']

share_config = {'class_names': class_names,
                'thre_per_classes': thre_per_classes,
                'min_img_size': 800,
                'num_classes': len(class_names),
                'aspect_ratios': (1., ),
                'anchor_sizes': (64, 128, 256, 512, 1024),
                'anchor_strides': (8, 16, 32, 64, 128),
                'octave': 2.,
                'scales_per_octave': 1,
                'straddle_thre': 0,
                'tower_dcn': False,
                'prior_prob': 0.01,
                'loss_gamma': 2.,
                'loss_alpha': 0.25,
                'pos_thre': 0.5,
                'neg_thre': 0.4,
                'top_k': 9,
                'reg_loss_alpha': 2.,
                'test_conf_thre': 0.05,
                'pre_nms_top_n': 1000,
                'nms_thre': 0.6,
                'post_nms_top_n': 100,
                'pixel_mean': [102.9801, 115.9465, 122.7717],
                'pixel_std': [1.0, 1.0, 1.0],
                'size_divisibility': 32,
                'max_iter': 90000,
                'multi_scale_range': (-1, -1),
                }


class dict2class:
    def __init__(self, config):
        for k, v in config.items():
            self.__setattr__(k, v)

    def print_cfg(self):
        print('\n' + '-' * 50)
        for k, v in vars(self).items():
            print(f'{k}: {v}')
        print()


def update_config(args=None):
    share_config.update(vars(args))
    return dict2class(share_config)
