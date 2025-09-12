# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmrotate.core import rbbox2roi, obb2xyxy, rbbox2result
from ..builder import (ROTATED_HEADS, build_roi_extractor, build_head)
import torch.nn.functional as F
from .oriented_standard_roi_head import OrientedStandardRoIHead

import os
import time
import cv2
import shutil
from mmcv.cnn import ConvModule

@ROTATED_HEADS.register_module()
class RotatedFIRoIHead(OrientedStandardRoIHead):
    """Simplest base roi head including one bbox head and one mask head."""
    def __init__(self,
                 roi_size=7,
                 num_gpus=1,
                 num_con_queue=256,
                 num_save_feats=300,
                 enc_output_dim=512,
                 proj_output_dim=128,
                 temperature=0.07,
                 ins_quality_assess_cfg=dict(
                     cls_score=0.00,
                     hq_score=0.01,
                     lq_score=0.005,
                     hq_pro_counts_thr=2),
                 con_sampler_cfg=dict(
                     num=128,
                     pos_fraction=[0.5, 0.25, 0.125]),
                 con_queue_dir=None,
                 num_classes=15,
                 iq_loss_weights=[0.5, 0.1, 0.05],
                 contrast_loss_weights=0.5,
                 hq_gt_aug_cfg=dict(
                     trans_range=[0.3, 0.5],
                     trans_num=2,
                     rescale_range=[0.97, 1.03],
                     rescale_num=2),
                 aug_roi_extractor=None,
                 init_cfg=dict(type='Normal', std=0.01,
                               override=[dict(name='fc_enc'), dict(name='fc_proj')]),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 *args,
                 **kwargs):
        super(RotatedFIRoIHead, self).__init__(
            *args, init_cfg=init_cfg, **kwargs)
        self.roi_size = roi_size
        self.num_gpus = num_gpus
        self.num_con_queue = num_con_queue
        self.num_save_feats = num_save_feats
        assert self.num_con_queue >= con_sampler_cfg['num']
        self.con_sampler_cfg = con_sampler_cfg
        self.con_sample_num = self.con_sampler_cfg['num']
        self.temperature = temperature
        self.iq_cls_score = ins_quality_assess_cfg['cls_score']
        self.hq_score = ins_quality_assess_cfg['hq_score']
        self.lq_score = ins_quality_assess_cfg['lq_score']
        self.hq_pro_counts_thr = ins_quality_assess_cfg['hq_pro_counts_thr']
        self.hq_gt_aug_cfg = hq_gt_aug_cfg
        # if self.training:
        #     self._mkdir(con_queue_dir, num_gpus)
        self.con_queue_dir = con_queue_dir
        self.num_classes = num_classes
        if aug_roi_extractor is None:
            aug_roi_extractor = dict(
                type='SingleRoIExtractor',
                roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32])
        self.aug_roi_extractor = build_roi_extractor(aug_roi_extractor)

        enc_input_dim = self.bbox_roi_extractor.out_channels  # roi_size ** 2 * self.bbox_roi_extractor.out_channels
        self.fc_enc = self._init_fc_enc(enc_input_dim, enc_output_dim)
        self.fc_proj = nn.Linear(enc_output_dim, proj_output_dim)
        self.relu = nn.ReLU(inplace=False)
        self.iq_loss_weights = iq_loss_weights
        self.contrast_loss_weights = contrast_loss_weights
        self.comp_convs = self._add_comp_convs(self.bbox_roi_extractor.out_channels,
                             roi_size, norm_cfg, act_cfg=None)
        self.mask_head = None
        
    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task. Always
                set to None.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox:

            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            assign_results = []
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])

                if gt_bboxes[i].numel() == 0:
                    sampling_result.pos_gt_bboxes = gt_bboxes[i].new(
                        (0, gt_bboxes[0].size(-1))).zero_()
                else:
                    sampling_result.pos_gt_bboxes = \
                        gt_bboxes[i][sampling_result.pos_assigned_gt_inds, :]

                assign_results.append(assign_result)
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, assign_results, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])

        return losses

    def _bbox_forward(self, x, rois):
        """Box head forward function used in both training and testing."""
        # TODO: a more flexible way to decide which feature maps to use
        bbox_feats = self.bbox_roi_extractor(
            x[:self.bbox_roi_extractor.num_inputs], rois)
        if self.with_shared_head:
            bbox_feats = self.shared_head(bbox_feats)
        feat_weights = bbox_feats.clone()
        for conv in self.comp_convs:
            feat_weights = conv(feat_weights)
        comp_feats = feat_weights.clone()
        feat_weights = F.softmax(feat_weights, dim=1)
        _, c, h, w = bbox_feats.size()
        weights = feat_weights.view(_, c, 1, 1).repeat(1, 1, h, w) + 1
        bbox_feats = bbox_feats * weights
        cls_score, bbox_pred = self.bbox_head(bbox_feats)

        bbox_results = dict(
            cls_score=cls_score, bbox_pred=bbox_pred, bbox_feats=bbox_feats, comp_feats=comp_feats)
        return bbox_results

    def _bbox_forward_train(self, x, assign_results, sampling_results, gt_bboxes, gt_labels,
                            img_metas):
        """Run forward function and calculate loss for box head in training.

        Args:
            x (list[Tensor]): list of multi-level img features.
            sampling_results (list[Tensor]): list of sampling results.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 5) in [cx, cy, w, h, a] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.

        Returns:
            dict[str, Tensor]: a dictionary of bbox_results.
        """
        rois = rbbox2roi([res.bboxes for res in sampling_results])
        bbox_results = self._bbox_forward(x, rois)

        bbox_targets = self.bbox_head.get_targets(sampling_results, gt_bboxes,
                                                  gt_labels, self.train_cfg)
        loss_bbox = self.bbox_head.loss(bbox_results['cls_score'],
                                        bbox_results['bbox_pred'], rois,
                                        *bbox_targets)

        # 按图像对cls_scores, bbox_feats, comp_feats进行分割
        num_proposals = [torch.sum(rois[:, 0] == i) for i in range(len(img_metas))]
        cls_scores = bbox_results['cls_score'].clone().split(num_proposals)
        bbox_feats = bbox_results['bbox_feats'].clone().split(num_proposals)
        comp_feats = bbox_results['comp_feats'].clone().split(num_proposals)  # [bs, num_proposals, 256, 1, 1]
        proposal_labels = bbox_targets[0].clone().split(num_proposals)

        # 初始化对比损失
        con_losses = cls_scores[0].new_zeros(1)

        # batchsize = len(sampling_results), 这里是逐图处理
        for i in range(len(sampling_results)):
            # 当前图像的真实框数量
            num_gts = len(gt_labels[i])
            if num_gts >=512:
                continue
            # 获取正样本的类别ID（去重）
            cat_ids = list(set(sampling_results[i].pos_gt_labels.tolist()))
            # 确定负样本采样数量（不超过设定最大值）
            cur_sample_num = min(sampling_results[i].neg_inds.size(0), self.con_sample_num)
            if num_gts == 0:
                contrast_loss = cls_scores[i].new_zeros(1)
                con_losses = con_losses + contrast_loss
                continue
            # 计算真实框的IQ_score
            iq_scores, pro_counts = self._ins_quality_assess(
                cls_scores[i],
                assign_results[i],
                sampling_results[i])
            # 加载高质量ROI特征和标签
            hq_feats, hq_labels = self._load_hq_roi_feats(bbox_feats[i], gt_labels[i], cat_ids)
            with torch.no_grad():
                for conv in self.comp_convs:
                    hq_feats = conv(hq_feats)  # [num_proposals, 256, 1, 1]
            con_roi_feats = torch.cat([comp_feats[i], hq_feats], dim=0)  # [num_proposals + num_hq, 256, 1, 1]
            hq_inds = (torch.nonzero((iq_scores >= self.hq_score) & \
                                    (pro_counts >= self.hq_pro_counts_thr),
                                    as_tuple=False))
            hq_inds = hq_inds.view(-1) # (N, )
            if len(hq_inds) == 0:    # no high-quality gt in current image
                aug_gt_ind = -1 * torch.ones(con_roi_feats.size(0))
                aug_num_per_hq_gt = 0
                aug_hq_gt_bboxes = gt_bboxes[i].new_empty(0)
                aug_gt_labels = gt_labels[i].new_empty(0)
            else:
                hq_gt_bboxes = sampling_results[i].pos_gt_bboxes[hq_inds]
                img_size = img_metas[i]['img_shape'][0]  # use img_w only since img_w == img_h
                aug_hq_gt_bboxes, aug_num_per_hq_gt = \
                    self._aug_hq_gt_bboxes(hq_gt_bboxes, img_size, img_size)
                aug_hq_gt_rois = rbbox2roi([aug_hq_gt_bboxes])
                aug_hq_gt_roi_feats = self.aug_roi_extractor(x, aug_hq_gt_rois)
                with torch.no_grad():
                    for conv in self.comp_convs:
                        aug_hq_gt_roi_feats = conv(aug_hq_gt_roi_feats)
                aug_gt_ind = hq_inds.view(-1, 1).repeat(1, aug_num_per_hq_gt).view(1, -1).squeeze(0)
                aug_gt_ind = torch.cat(
                    [-1 * aug_gt_ind.new_ones(con_roi_feats.size(0)), aug_gt_ind], dim=-1)
                aug_gt_labels = sampling_results[i].pos_gt_labels[hq_inds].view(
                    -1, 1).repeat(1, aug_num_per_hq_gt).view(1, -1).squeeze(0)
                con_roi_feats = torch.cat([con_roi_feats, aug_hq_gt_roi_feats], dim=0)  # [num_proposals + num_hq + num_hq_aug, 256, 1, 1]
            iq_signs, ex_pos_nums = self._get_gt_quality(
                iq_scores, aug_num_per_hq_gt, gt_labels[i], cur_sample_num)
            is_hq = torch.cat(
                [gt_labels[i].new_zeros(num_proposals[i]),
                 torch.ones_like(hq_labels),
                 -gt_labels[i].new_ones(aug_hq_gt_bboxes.size(0))], dim=-1)
            roi_labels = torch.cat(
                [proposal_labels[i], hq_labels, aug_gt_labels], dim=-1)
            assert roi_labels.size(0) == con_roi_feats.size(0)
            # for dense ground-truth situation, only a part of gt will be processed,
            # which resembles the way of gt being handled in bbox_sampler
            num_actual_gts = sampling_results[i].pos_is_gt.sum()
            pos_assigned_gt_inds = sampling_results[i].pos_assigned_gt_inds
            pos_is_gt = sampling_results[i].pos_is_gt.bool()
            pos_assigned_actual_gt_inds = pos_assigned_gt_inds[pos_is_gt]
            iq_scores = iq_scores[pos_assigned_actual_gt_inds]
            iq_signs = iq_signs[pos_assigned_actual_gt_inds]
            ex_pos_nums = ex_pos_nums[pos_assigned_actual_gt_inds]
            labels = gt_labels[i][pos_assigned_actual_gt_inds]
            sample_inds, pos_signs = self._sample(
                iq_signs, ex_pos_nums, labels, roi_labels, is_hq, aug_gt_ind, cur_sample_num)
            if sample_inds.size(1) == 0:  # 检查第二维度是否为空
                b = True
            # anchor_feature: (num_gts, 256, 7, 7)
            # contrast_feature: (num_gts, self.con_sample_num, 256, 7, 7)
            anchor_feature = con_roi_feats[:num_actual_gts]
            contrast_feature = con_roi_feats[sample_inds]
            assert anchor_feature.size(0) == contrast_feature.size(0)
            iq_loss_weights = torch.ones_like(iq_scores)
            for j, weight in enumerate(self.iq_loss_weights):
                cur_signs = torch.nonzero(iq_signs == j).view(-1)
                iq_loss_weights[cur_signs] = weight * iq_loss_weights[cur_signs]
            loss = self.contrast_forward(anchor_feature, contrast_feature,
                                         pos_signs, iq_loss_weights)
            contrast_loss = self.contrast_loss_weights * loss
            con_losses = con_losses + contrast_loss

            # save high-quality features at last
            # for dense ground-truth situation
            pro_counts = pro_counts[pos_assigned_actual_gt_inds]
            hq_inds = torch.nonzero((iq_scores >= self.hq_score) & \
                                    (pro_counts >= self.hq_pro_counts_thr),
                                    as_tuple=False).view(-1)  # (N, )
            # high-quality proposals: high instance quality scores and
            # sufficient numbers of proposals
            if len(hq_inds) > 0:
                hq_scores, hq_pro_counts = \
                    iq_scores[hq_inds], pro_counts[hq_inds]
                for hq_score, hq_pro_count, hq_gt_ind in \
                        zip(hq_scores, hq_pro_counts, hq_inds):
                    cur_gt_cat_id = sampling_results[i].pos_gt_labels[hq_gt_ind]
                    cur_gt_roi_feat = bbox_feats[i][hq_gt_ind, :, :, :].clone()
                    self._update_iq_score_info(cur_gt_cat_id.item(), cur_gt_roi_feat)
        if len(con_losses) > 0:
            con_loss = con_losses / len(assign_results)
            loss_bbox.update(loss_con=con_loss)

        bbox_results.update(loss_bbox=loss_bbox)
        return bbox_results
    
    def _ins_quality_assess(self, cls_score, assign_result, sampling_result,
                            eps=1e-6):
        """ Compute the quality of instances in a single image
            The quality of an instance is defined:
                iq = 1 / N * (IoU * Score)_i (i: {1, 2, ..., N})
        """
        with torch.no_grad():
            num_gts = sampling_result.num_gts
            assign_pos_inds = sampling_result.pos_inds
            num_pos = len(assign_pos_inds)
            pos_gt_labels = sampling_result.pos_gt_labels
            scores = F.softmax(cls_score[:num_pos, :], dim=-1)
            scores = torch.gather(
                scores, dim=1, index=pos_gt_labels.view(-1, 1)).view(-1)  # (num_pos, )
            iq_candi_inds = scores >= self.iq_cls_score
            if torch.sum(iq_candi_inds) == 0:
                return scores.new_zeros(num_gts), scores.new_zeros(num_gts)
            else:
                scores = scores[iq_candi_inds]
                num_pos = len(scores)
                pos_ious = assign_result.max_overlaps[assign_pos_inds[iq_candi_inds]]  # (num_pos, )
                pos_is_pro = (sampling_result.pos_is_gt == 0)[iq_candi_inds]  # (num_pos, )
                pos_assigned_gt_inds = sampling_result.pos_assigned_gt_inds[iq_candi_inds]  # (num_pos, )
                gt_ind_mask = torch.cat([pos_assigned_gt_inds == i for i in range(num_gts)]
                                        ).contiguous().view(num_gts, num_pos)
                # compute proposals (ious and scores) only
                # TODO: enusre the return length is num_gts
                iq = pos_ious * pos_is_pro * gt_ind_mask * scores  # (num_gts, num_pos)
                iq_sum = torch.sum(iq, dim=1)  # (num_gts, )
                iq_count = torch.sum(gt_ind_mask * pos_is_pro, dim=1)  # (num_gts, )
                iq_count_eps = iq_count + eps * (iq_count == 0)
                iq_score = torch.div(iq_sum, iq_count_eps)
                return iq_score, iq_count
            
    def _update_iq_score_info(self, cat_id, cur_gt_roi_feat):
        cur_gt_roi_feat = cur_gt_roi_feat.view(-1, 256, 7, 7)
        # update the iq_score queue and corresponding dict info
        device_dir = str(cur_gt_roi_feat.device.index)
        cur_gt_save_pth = os.path.join(
            self.con_queue_dir, device_dir, str(cat_id) + '.pt')
        if os.path.exists(cur_gt_save_pth):
            cur_pt = torch.load(cur_gt_save_pth).view(-1, 256, 7, 7)
            os.remove(cur_gt_save_pth)
            cur_gt_roi_feat = torch.cat(
                [cur_pt.to(cur_gt_roi_feat.device), cur_gt_roi_feat], dim=0)
        cur_gt_roi_feat = cur_gt_roi_feat.view(-1, 256, 7, 7)
        dup_len = cur_gt_roi_feat.size(0) > int(self.num_con_queue // self.num_gpus)
        if dup_len > 0:
            cur_gt_roi_feat = cur_gt_roi_feat[-dup_len, ...]
        torch.save(
            cur_gt_roi_feat, cur_gt_save_pth, _use_new_zipfile_serialization=False)

    def _load_hq_roi_feats(self, roi_feats, gt_labels, cat_ids):
        device_id = str(gt_labels.device.index)  # current GPU id
        with torch.no_grad():
            hq_feats, hq_labels = [], []
            for cat_id in range(self.num_classes):
                if cat_id not in cat_ids:
                    continue
                cur_cat_feat_pth = os.path.join(
                    self.con_queue_dir, device_id, str(cat_id) + '.pt')
                cur_cat_feat = torch.load(cur_cat_feat_pth) \
                    if os.path.exists(cur_cat_feat_pth) \
                    else roi_feats.new_empty(0)
                cur_cat_roi_feats = cur_cat_feat.to(roi_feats.device).view(-1, 256, 7, 7)
                cur_hq_labels = cat_id * gt_labels.new_ones(
                    cur_cat_roi_feats.size(0)).to(gt_labels.device)
                hq_feats.append(cur_cat_roi_feats)
                hq_labels.append(cur_hq_labels)
            hq_feats = torch.as_tensor(
                torch.cat(hq_feats, dim=0),
                dtype=roi_feats.dtype).view(-1, 256, 7, 7)
            hq_labels = torch.as_tensor(
                torch.cat(hq_labels, dim=-1), dtype=gt_labels.dtype)
        return hq_feats, hq_labels

    def _aug_hq_gt_bboxes(self, hq_gt_bboxes, img_w, img_h):
        with torch.no_grad():
            hq_gt_bboxes = hq_gt_bboxes.view(-1, 5)  # 调整为 (num_gts, 5)
            num_gts = hq_gt_bboxes.size(0)
            trans_range, rescale_range = \
                self.hq_gt_aug_cfg['trans_range'], self.hq_gt_aug_cfg['rescale_range']
            trans_num, rescale_num = \
                self.hq_gt_aug_cfg['trans_num'], self.hq_gt_aug_cfg['rescale_num']

            # 生成平移和缩放系数
            trans_ratios = torch.linspace(
                trans_range[0], trans_range[1], trans_num, device=hq_gt_bboxes.device)
            rescale_ratios = torch.linspace(
                rescale_range[0], rescale_range[1], rescale_num, device=hq_gt_bboxes.device)

            # 平移增强逻辑
            gt_bboxes = hq_gt_bboxes.unsqueeze(1)  # (num_gts, 1, 5)
            trans_candi = gt_bboxes.repeat(1, 4 * trans_num, 1)  # (num_gts, 4*trans_num, 5)

            # 计算四个方向的平移量 (dx, dy)
            w = hq_gt_bboxes[:, 2]  # 原始宽度
            h = hq_gt_bboxes[:, 3]  # 原始高度
            wh = torch.cat([w.view(-1, 1), h.view(-1, 1)], dim=1).unsqueeze(1)  # (num_gts, 1, 2)
            inter_mat = torch.tensor([[1, 0], [0, 1], [-1, 0], [0, -1]], device=hq_gt_bboxes.device)
            wh_mat = wh * inter_mat  # (num_gts, 4, 2)
            scaled_wh = torch.cat(  # (num_gts, 4*trans_num, 2)
                [r * wh_mat for r in trans_ratios], dim=1)
            trans_wh = torch.zeros((num_gts, 4 * trans_num, 5), device=scaled_wh.device)  # 初始化五维偏移量
            trans_wh[..., :2] = scaled_wh  # 仅前两维（cx, cy）应用平移
            trans_gt_bboxes = trans_candi.clone()
            trans_gt_bboxes[..., :2] += trans_wh[..., :2]  # 仅更新中心点坐标
            trans_gt_bboxes = torch.clamp(trans_gt_bboxes, 0, img_w)

            # 缩放增强逻辑
            rescaled_gt_bboxes = self.rescale_gt_bboxes(
                hq_gt_bboxes, rescale_ratios)  # (num_gts, rescale_num, 5)
            rescaled_gt_bboxes = torch.clamp(rescaled_gt_bboxes, 0, img_w)

            # 合并增强结果
            aug_gt_bboxes = []
            for i in range(num_gts):
                aug_gt_bboxes.append(
                    torch.cat([trans_gt_bboxes[i], rescaled_gt_bboxes[i]],
                              dim=0))
            aug_gt_bboxes = torch.cat(aug_gt_bboxes, dim=0)  # (num_gts, 4*trans_num+rescale_num, 4)
            aug_num_per_hq_gt = 4 * trans_num + rescale_num

        return aug_gt_bboxes, aug_num_per_hq_gt

    def rescale_gt_bboxes(self, gt_bboxes, scale_factors):
        cx = gt_bboxes[:, 0]
        cy = gt_bboxes[:, 1]
        w = gt_bboxes[:, 2]
        h = gt_bboxes[:, 3]
        a = gt_bboxes[:, 4]
        rescaled_gt_bboxes = []
        for scale_factor in scale_factors:
            new_w = w * scale_factor
            new_h = h * scale_factor
            rescaled_gt_bboxes.append(
                torch.stack((cx, cy, new_w, new_h, a), dim=-1))
        rescaled_gt_bboxes = torch.cat(
            rescaled_gt_bboxes, dim=0).view(gt_bboxes.size(0), -1, 5)
        return rescaled_gt_bboxes
    
    def _get_gt_quality(self, iq_scores, aug_num_per_hq_gt, gt_labels, cur_sample_num):
        """ low-quality:  0;
            mid_qulity:   1;
            high-quality: 2;
        """
        with torch.no_grad():
            iq_signs = torch.zeros_like(iq_scores)  # low-quality
            iq_signs[iq_scores >= self.lq_score] = 1  # mid-quality
            iq_signs[iq_scores >= self.hq_score] = 2  # high-quality
            pos_fraction = self.con_sampler_cfg['pos_fraction']
            ex_pos_nums = gt_labels.new_ones(iq_scores.size(0))
            for val in range(2):
                ex_pos_nums[iq_signs == val] = int(cur_sample_num * pos_fraction[val])
            ex_pos_nums[iq_signs == 2] = aug_num_per_hq_gt
        return iq_signs, ex_pos_nums
    
    def _sample(self, iq_signs, ex_pos_nums, gt_labels, roi_labels,
                is_hq, aug_gt_ind, cur_sample_num):
        """
        Returns:
            sample_inds : indices of pos and neg samples (num_gts, self.con_sample_num)
            pos_signs   : whether the sample of current index is positive
        """
        sample_inds, pos_signs = [], []
        for gt_ind in range(len(gt_labels)):
            ex_pos_num = ex_pos_nums[gt_ind]
            iq_sign = iq_signs[gt_ind]
            # sample positives first
            if iq_sign == 2:
                pos_inds = torch.nonzero(aug_gt_ind == gt_ind, as_tuple=False).view(-1)
            else:
                can_pos_inds = torch.nonzero(
                    (is_hq == 1) & (roi_labels == gt_labels[gt_ind]),
                    as_tuple=False).view(-1)
                if len(can_pos_inds) <= ex_pos_num:
                    pos_inds = can_pos_inds
                else:
                    pos_inds = self._random_choice(can_pos_inds, ex_pos_num)
            # sample negatives then
            can_neg_inds = torch.nonzero(
                (roi_labels != gt_labels[gt_ind]) & (is_hq == 0),
                as_tuple=False).view(-1)
            neg_inds = self._random_choice(
                can_neg_inds, cur_sample_num - len(pos_inds))
            sample_inds.append(
                torch.cat([pos_inds.cuda(), neg_inds.cuda()], dim=-1).view(1, -1))
            pos_signs.append(
                torch.cat([torch.ones_like(pos_inds.cuda()),
                           torch.zeros_like(neg_inds.cuda())], dim=-1).view(1, -1))
        sample_inds = torch.cat(sample_inds, dim=0)
        pos_signs = torch.cat(pos_signs, dim=0)
        return sample_inds, pos_signs

    def _random_choice(self, gallery, num):
        # fork from RandomSampler
        assert len(gallery) >= num
        is_tensor = isinstance(gallery, torch.Tensor)
        if not is_tensor:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
            else:
                device = 'cpu'
            gallery = torch.tensor(gallery, dtype=torch.long, device=device)
        perm = torch.randperm(gallery.numel())[:num].to(device=gallery.device)
        rand_inds = gallery[perm]
        if not is_tensor:
            rand_inds = rand_inds.cpu().numpy()
        return rand_inds

    def _mkdir(self, con_queue_dir, num_gpus):
        if os.path.exists(con_queue_dir):
            shutil.rmtree(con_queue_dir)
        os.mkdir(con_queue_dir)
        for i in range(num_gpus):
            os.makedirs(os.path.join(con_queue_dir, str(i)))

    def _init_fc_enc(self, enc_input_dim, enc_output_dim):
        fc_enc = nn.ModuleList()
        fc_enc.append(nn.Linear(enc_input_dim, enc_output_dim))
        fc_enc.append(nn.Linear(enc_output_dim, enc_output_dim))
        return fc_enc

    def _add_comp_convs(self, in_channels, roi_feat_size, norm_cfg, act_cfg):
        comp_convs = nn.ModuleList()
        for i in range(roi_feat_size//2):
            comp_convs.append(
                ConvModule(in_channels, in_channels, 3, norm_cfg=norm_cfg, act_cfg=act_cfg)
            )
        return comp_convs
    
    def contrast_forward(self, anchor_feature, contrast_feature,
                         pos_signs, loss_weights, eps=1e-6):
        """
        Args:
            anchor_feature: ground-truth roi features in a single image
                (num_gts, 256, 1, 1)
            contrast_feature: pos/neg rois features fro training
                (num_gts, self.con_sample_num, 256, 1, 1)
            pos_signs: indicate whether the sample pos/neg (1/0)
                (num_gts, self.con_sample_num)
            loss_weights: loss weights of each gt (num_gts, )
        """
        anchor_feature = anchor_feature.view(anchor_feature.size()[:-2])  # [num_gts, 256]
        contrast_feature = contrast_feature.view(contrast_feature.size()[:-2])  # [num_gts, self.con_sample_num, 256]
        for fc in self.fc_enc:
            anchor_feature = self.relu(fc(anchor_feature))
            contrast_feature = self.relu(fc(contrast_feature))
        anchor_feature = self.fc_proj(anchor_feature)
        contrast_feature = self.fc_proj(contrast_feature)
        anchor_feats = F.normalize(anchor_feature, dim=-1)  # (num_gts, 128)
        contrast_feats = F.normalize(contrast_feature, dim=-1)  # (num_gts, self.con_sample_num, 128)
        sim_logits = torch.div(  # (num_gts, self.con_sample_num)
            torch.matmul(anchor_feats.unsqueeze(1),
                         contrast_feats.transpose(2, 1).contiguous()),
            self.temperature).squeeze(1)
        # for numerical stability
        sim_logits_max, _ = torch.max(sim_logits, dim=1, keepdim=True)
        logits = sim_logits - sim_logits_max.detach()  # (num_gts, self.con_sample_num)

        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        pos_num = pos_signs.sum(dim=1).cuda()
        pos_num = pos_num + eps * (pos_num == 0)  # avoid dividing by zero
        mean_log_prob_pos = -(pos_signs * log_prob).sum(dim=1) / pos_num
        weighted_loss = loss_weights * mean_log_prob_pos
        loss = weighted_loss.mean()
        return loss

    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    # gt_bboxes, gt_labels,
                    proposals=None,
                    rescale=False):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (batch_size, c, h, w).
            proposal_list (list(Tensor)): Proposals from rpn head.
                Each has shape (num_proposals, 5), last dimension
                5 represent (x1, y1, x2, y2, score).
            img_metas (list[dict]): Meta information of images.
            rescale (bool): Whether to rescale the results to
                the original image. Default: True.

        Returns:
            list[list[np.ndarray]] or list[tuple]: When no mask branch,
            it is bbox results of each image and classes with type
            `list[list[np.ndarray]]`. The outer list
            corresponds to each image. The inner list
            corresponds to each class. When the model has mask branch,
            it contains bbox results and mask results.
            The outer list corresponds to each image, and first element
            of tuple is bbox results, second element is mask results.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)
        bbox_results = [
            rbbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        #if not self.with_mask:
        return bbox_results
        # else:
        #     segm_results = self.simple_test_mask(
        #         x, img_metas, det_bboxes, det_labels, rescale=rescale)
        #     return list(zip(bbox_results, segm_results))
