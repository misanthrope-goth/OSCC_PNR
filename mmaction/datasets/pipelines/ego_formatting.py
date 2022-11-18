from collections.abc import Sequence

import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC

from ..builder import PIPELINES


@PIPELINES.register_module()
class FormatVideoShape:
    """Format final imgs shape to the given input_format.

    Required keys are "imgs", "num_clips" and "clip_len", added or modified
    keys are "imgs" and "input_shape".

    Args:
        input_format (str): Define the final imgs format.
        collapse (bool): To collpase input_format N... to ... (NCTHW to CTHW,
            etc.) if N is 1. Should be set as True when training and testing
            detectors. Default: False.
    """

    def __init__(self, input_format, collapse=False, sparse_sample=False, num_sparse_clips=1):
        self.input_format = input_format
        self.collapse = collapse
        self.sparse_sample = sparse_sample
        self.num_sparse_clips = num_sparse_clips
        if self.input_format not in ['NCTHW', 'NCHW', 'NCHW_Flow', 'NPTCHW']:
            raise ValueError(
                f'The input format {self.input_format} is invalid.')

    def __call__(self, results):
        """Performs the FormatShape formatting.

        Args:
            results (dict): The resulting dict to be modified and passed
                to the next transform in pipeline.
        """
        if not isinstance(results['imgs'], np.ndarray):
            results['imgs'] = np.array(results['imgs'])
        imgs = results['imgs']
        # [M x H x W x C]
        # M = 1 * N_crops * N_clips * L
        if self.collapse:
            assert results['num_clips'] == 1

        if self.sparse_sample:
            assert results['num_clips'] >= 1 and results['clip_len'] == 1 and results[
                'frame_interval'] == 1 and self.input_format == 'NCTHW'

        if self.input_format == 'NCTHW':
            num_clips = results['num_clips']
            clip_len = results['clip_len']
            imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
            # N_crops x N_clips x L x H x W x C

            if self.sparse_sample:
                imgs = imgs.reshape((-1, num_clips * clip_len) + imgs.shape[3:])
                sparse_clips = []
                for i in range(self.num_sparse_clips):
                    sc = imgs[:, i::self.num_sparse_clips]
                    sparse_clips.append(sc)
                sparse_clips = np.stack(sparse_clips, axis=1)  # N_crops x N_clips x L x H x W x C
                imgs = sparse_clips.reshape((-1,) + sparse_clips.shape[2:])

                # N_crops x N_clips x H x W x C
                imgs = np.transpose(imgs, (0, 4, 1, 2, 3))
            else:
                imgs = np.transpose(imgs, (0, 1, 5, 2, 3, 4))
                # N_crops x N_clips x C x L x H x W
                imgs = imgs.reshape((-1,) + imgs.shape[2:])
            # M' x C x L x H x W
            # M' = N_crops x N_clips
        elif self.input_format == 'NCHW':
            imgs = np.transpose(imgs, (0, 3, 1, 2))
            # M x C x H x W
        elif self.input_format == 'NCHW_Flow':
            num_clips = results['num_clips']
            clip_len = results['clip_len']

            imgs = imgs.reshape((-1, num_clips, clip_len) + imgs.shape[1:])
            # N_crops x N_clips x L x H x W x C
            imgs = np.transpose(imgs, (0, 1, 2, 5, 3, 4))
            # N_crops x N_clips x L x C x H x W
            imgs = imgs.reshape((-1, imgs.shape[2] * imgs.shape[3]) +
                                imgs.shape[4:])
            # M' x C' x H x W
            # M' = N_crops x N_clips
            # C' = L x C
        elif self.input_format == 'NPTCHW':
            num_proposals = results['num_proposals']
            num_clips = results['num_clips']
            clip_len = results['clip_len']
            imgs = imgs.reshape((num_proposals, num_clips * clip_len) +
                                imgs.shape[1:])
            # P x M x H x W x C
            # M = N_clips x L
            imgs = np.transpose(imgs, (0, 1, 4, 2, 3))
            # P x M x C x H x W

        if self.collapse:
            assert imgs.shape[0] == 1
            imgs = imgs.squeeze(0)

        results['imgs'] = imgs
        results['input_shape'] = imgs.shape
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(input_format='{self.input_format}')"
        return repr_str
