# modified from https://github.com/rowanz/neural-motifs
import torch
from torch import nn
from torch.nn import functional as F

from pysgg.data import get_dataset_statistics
from pysgg.modeling.make_layers import make_fc
from pysgg.modeling.roi_heads.relation_head.utils_relation import get_box_pair_info, get_box_info, \
    layer_init
from pysgg.modeling.utils import cat
from .utils_motifs import obj_edge_vectors, encode_box_info
from pysgg.data.datasets.visual_genome import load_info
from pysgg.utils.imports import import_file

class IMPContext(nn.Module):
    def __init__(self, config, in_channels, hidden_dim=512, num_iter=3):
        super(IMPContext, self).__init__()
        self.cfg = config

        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.hidden_dim = hidden_dim
        self.num_iter = num_iter

        self.pairwise_feature_extractor = PairwiseFeatureExtractor(config,
                                                                   in_channels)

        # mode
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        self.obj_unary = make_fc(self.pooling_dim, hidden_dim)
        self.edge_unary = make_fc(self.pooling_dim, hidden_dim)

        self.edge_gru = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)
        self.node_gru = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)

        self.sub_vert_w_fc = nn.Sequential(make_fc(hidden_dim * 2, 1), nn.Sigmoid())
        self.obj_vert_w_fc = nn.Sequential(make_fc(hidden_dim * 2, 1), nn.Sigmoid())
        self.out_edge_w_fc = nn.Sequential(make_fc(hidden_dim * 2, 1), nn.Sigmoid())
        self.in_edge_w_fc = nn.Sequential(make_fc(hidden_dim * 2, 1), nn.Sigmoid())
        # untreated average features
        self.average_ratio = 0.0005
        self.effect_analysis = config.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS
        # untreated average features
        self.effect_analysis = config.MODEL.ROI_RELATION_HEAD.CAUSAL.EFFECT_ANALYSIS
        if self.effect_analysis:
            self.embed_dim = self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM
            self.obj_dim = in_channels
            if self.cfg.MODEL.ROI_RELATION_HEAD.VISUAL_LANGUAGE_MERGER_OBJ:
                self.language_obj_dim=512
            else:
                self.language_obj_dim=0
            self.register_buffer("untreated_obj_feat",
                                 torch.zeros( self.obj_dim+self.language_obj_dim))
            self.register_buffer("untreated_edg_feat",
                                 torch.zeros( self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM))

    def moving_average(self, holder, input):
        assert len(input.shape) == 2
        with torch.no_grad():
            holder = holder * (1 - self.average_ratio) + self.average_ratio * input.mean(0).view(-1)
        return holder

    def forward(self, inst_features, proposals, union_features, rel_pair_idxs, logger=None,ctx_average=False):
        num_objs = [len(b) for b in proposals]
        batch_size = inst_features.shape[0]
        rel_batch_size = union_features.shape[0]
        if (not self.training) and self.effect_analysis and ctx_average:
            roi_input = self.untreated_obj_feat.view(1, -1).expand(batch_size, -1)
            uni_input = self.untreated_edg_feat.view(1, -1).expand(rel_batch_size, -1)

        else:
            roi_input=inst_features
            uni_input=union_features
        augment_obj_feat, rel_feats = self.pairwise_feature_extractor(roi_input, uni_input,
                                                                      proposals, rel_pair_idxs)

        obj_rep = self.obj_unary(augment_obj_feat)
        rel_rep = F.relu(self.edge_unary(rel_feats))

        obj_count = obj_rep.shape[0]
        rel_count = rel_rep.shape[0]

        # generate sub-rel-obj mapping
        sub2rel = torch.zeros(obj_count, rel_count).to(obj_rep.device).float()
        obj2rel = torch.zeros(obj_count, rel_count).to(obj_rep.device).float()
        obj_offset = 0
        rel_offset = 0
        sub_global_inds = []
        obj_global_inds = []
        for pair_idx, num_obj in zip(rel_pair_idxs, num_objs):
            num_rel = pair_idx.shape[0]
            sub_idx = pair_idx[:, 0].contiguous().long().view(-1) + obj_offset
            obj_idx = pair_idx[:, 1].contiguous().long().view(-1) + obj_offset
            rel_idx = torch.arange(num_rel).to(obj_rep.device).long().view(-1) + rel_offset

            sub_global_inds.append(sub_idx)
            obj_global_inds.append(obj_idx)

            sub2rel[sub_idx, rel_idx] = 1.0
            obj2rel[obj_idx, rel_idx] = 1.0

            obj_offset += num_obj
            rel_offset += num_rel

        sub_global_inds = torch.cat(sub_global_inds, dim=0)
        obj_global_inds = torch.cat(obj_global_inds, dim=0)

        # iterative message passing
        hx_obj = torch.zeros(obj_count, self.hidden_dim, requires_grad=False).to(obj_rep.device).float()
        hx_rel = torch.zeros(rel_count, self.hidden_dim, requires_grad=False).to(obj_rep.device).float()

        vert_factor = [self.node_gru(obj_rep, hx_obj)]
        edge_factor = [self.edge_gru(rel_rep, hx_rel)]

        for i in range(self.num_iter):
            # compute edge context
            sub_vert = vert_factor[i][sub_global_inds]
            obj_vert = vert_factor[i][obj_global_inds]
            weighted_sub = self.sub_vert_w_fc(
                torch.cat((sub_vert, edge_factor[i]), 1)) * sub_vert
            weighted_obj = self.obj_vert_w_fc(
                torch.cat((obj_vert, edge_factor[i]), 1)) * obj_vert

            edge_factor.append(self.edge_gru(weighted_sub + weighted_obj, edge_factor[i]))

            # Compute vertex context
            pre_out = self.out_edge_w_fc(torch.cat((sub_vert, edge_factor[i]), 1)) * edge_factor[i]
            pre_in = self.in_edge_w_fc(torch.cat((obj_vert, edge_factor[i]), 1)) * edge_factor[i]
            vert_ctx = sub2rel @ pre_out + obj2rel @ pre_in
            vert_factor.append(self.node_gru(vert_ctx, vert_factor[i]))
        if self.training and self.effect_analysis:
            self.untreated_obj_feat = self.moving_average(self.untreated_obj_feat, inst_features)
            self.untreated_edg_feat = self.moving_average(self.untreated_edg_feat, union_features)
        return vert_factor[-1], edge_factor[-1]


class PairwiseFeatureExtractor(nn.Module):
    """
    extract the pairwise features from the object pairs and union features.
    most pipeline keep same with the motifs instead the lstm massage passing process
    """

    def __init__(self, config, in_channels):
        super(PairwiseFeatureExtractor, self).__init__()
        self.cfg = config
        paths_catalog = import_file(
            "pysgg.config.paths_catalog", self.cfg.PATHS_CATALOG, True
        )
        dataset_names = self.cfg.DATASETS.TRAIN
        DatasetCatalog = paths_catalog.DatasetCatalog
        for dataset_name in dataset_names:
            data = DatasetCatalog.get(dataset_name, self.cfg)
            dict_file = data['args']['dict_file']
        self.obj_classes, self.rel_classes, self.ind_to_attributes = load_info(
            dict_file)

        self.num_obj_classes = len(self.obj_classes)
        self.num_rel_classes = len(self.rel_classes)

        # mode
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        # features augmentation for instance features
        # word embedding
        # add language prior representation according to the prediction distribution
        # of objects
        self.embed_dim = self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        self.obj_dim = in_channels
        self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM

        self.word_embed_feats_on = self.cfg.MODEL.ROI_RELATION_HEAD.WORD_EMBEDDING_FEATURES
        if self.word_embed_feats_on:
            obj_embed_vecs = obj_edge_vectors(self.obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)##[151,200]
            self.obj_embed_on_prob_dist = nn.Embedding(self.num_obj_classes, self.embed_dim)
            self.obj_embed_on_pred_label = nn.Embedding(self.num_obj_classes, self.embed_dim)
            with torch.no_grad():
                self.obj_embed_on_prob_dist.weight.copy_(obj_embed_vecs, non_blocking=True)
                self.obj_embed_on_pred_label.weight.copy_(obj_embed_vecs, non_blocking=True)
        else:
            self.embed_dim = 0

        # features augmentation for rel pairwise features
        self.rel_feature_type = config.MODEL.ROI_RELATION_HEAD.EDGE_FEATURES_REPRESENTATION#fusion

        # the input dimension is ROI head MLP, but the inner module is pooling dim, so we need
        # to decrease the dimension first.
        if self.pooling_dim != in_channels:
            self.rel_feat_dim_not_match = True
            self.rel_feature_up_dim = make_fc(in_channels, self.pooling_dim)
            layer_init(self.rel_feature_up_dim, xavier=True)
        else:
            self.rel_feat_dim_not_match = False
        if self.cfg.MODEL.ROI_RELATION_HEAD.VISUAL_LANGUAGE_MERGER_OBJ:
            self.pairwise_obj_feat_updim_fc = make_fc(self.hidden_dim + self.obj_dim + self.embed_dim+512,
                                                  self.hidden_dim * 2)
        else:
            self.pairwise_obj_feat_updim_fc = make_fc(self.hidden_dim + self.obj_dim + self.embed_dim,
                                                      self.hidden_dim * 2)

        self.outdim = self.pooling_dim
        # position embedding
        # encode the geometry information of bbox in relationships
        self.geometry_feat_dim = 128
        self.pos_embed = nn.Sequential(*[
            make_fc(9, 32), nn.BatchNorm1d(32, momentum=0.001),
            make_fc(32, self.geometry_feat_dim), nn.ReLU(inplace=True),
        ])

        if self.rel_feature_type in ["obj_pair", "fusion"]:
            self.spatial_for_vision = config.MODEL.ROI_RELATION_HEAD.CAUSAL.SPATIAL_FOR_VISION
            if self.spatial_for_vision:
                self.spt_emb = nn.Sequential(*[make_fc(32, self.hidden_dim),
                                               nn.ReLU(inplace=True),
                                               make_fc(self.hidden_dim, self.hidden_dim * 2),
                                               nn.ReLU(inplace=True)
                                               ])
                layer_init(self.spt_emb[0], xavier=True)
                layer_init(self.spt_emb[2], xavier=True)

            self.pairwise_rel_feat_finalize_fc = nn.Sequential(
                make_fc(self.hidden_dim * 2, self.pooling_dim),
                nn.ReLU(inplace=True),
            )

        # map bidirectional hidden states of dimension self.hidden_dim*2 to self.hidden_dim
        if self.cfg.MODEL.ROI_RELATION_HEAD.VISUAL_LANGUAGE_MERGER_OBJ:
            self.obj_hidden_linear = make_fc(self.obj_dim + self.embed_dim + self.geometry_feat_dim+512, self.hidden_dim)

            self.obj_feat_aug_finalize_fc = nn.Sequential(
                make_fc(self.hidden_dim + self.obj_dim + self.embed_dim+512, self.pooling_dim),
                nn.ReLU(inplace=True),
            )
        else:
            self.obj_hidden_linear = make_fc(self.obj_dim + self.embed_dim + self.geometry_feat_dim,
                                             self.hidden_dim)

            self.obj_feat_aug_finalize_fc = nn.Sequential(
                make_fc(self.hidden_dim + self.obj_dim + self.embed_dim, self.pooling_dim),
                nn.ReLU(inplace=True),
            )



    def pairwise_rel_features(self, augment_obj_feat, union_features, rel_pair_idxs, inst_proposals):
        obj_boxs = [get_box_info(p.bbox, need_norm=True, proposal=p) for p in inst_proposals]
        num_objs = [len(p) for p in inst_proposals]
        # post decode
        # (num_objs, hidden_dim) -> (num_objs, hidden_dim * 2)
        # going to split single object representation to sub-object role of relationship
        pairwise_obj_feats_fused = self.pairwise_obj_feat_updim_fc(augment_obj_feat)
        pairwise_obj_feats_fused = pairwise_obj_feats_fused.view(pairwise_obj_feats_fused.size(0), 2, self.hidden_dim)
        head_rep = pairwise_obj_feats_fused[:, 0].contiguous().view(-1, self.hidden_dim)
        tail_rep = pairwise_obj_feats_fused[:, 1].contiguous().view(-1, self.hidden_dim)
        # split 对应每个图片的head_rep
        head_reps = head_rep.split(num_objs, dim=0)
        tail_reps = tail_rep.split(num_objs, dim=0)
        # generate the pairwise object for relationship representation
        # (num_objs, hidden_dim) <rel pairing > (num_objs, hidden_dim)
        #   -> (num_rel, hidden_dim * 2)
        #   -> (num_rel, hidden_dim)
        obj_pair_feat4rel_rep = []
        pair_bboxs_info = []

        for pair_idx, head_rep, tail_rep, obj_box in zip(rel_pair_idxs, head_reps, tail_reps, obj_boxs):
            obj_pair_feat4rel_rep.append(torch.cat((head_rep[pair_idx[:, 0]], tail_rep[pair_idx[:, 1]]), dim=-1))#把head tail特征拼一行
            pair_bboxs_info.append(get_box_pair_info(obj_box[pair_idx[:, 0]], obj_box[pair_idx[:, 1]]))#get_box_pair_info这个包括的是pair的相互位置关系，最后加在特征上
        pair_bbox_geo_info = cat(pair_bboxs_info, dim=0)#根据坐标得到的box编码
        obj_pair_feat4rel_rep = cat(obj_pair_feat4rel_rep, dim=0)  # (num_rel, hidden_dim * 2)
        if self.spatial_for_vision:#default true
            obj_pair_feat4rel_rep = obj_pair_feat4rel_rep * self.spt_emb(pair_bbox_geo_info)#visual representation和位置编码相乘

        obj_pair_feat4rel_rep = self.pairwise_rel_feat_finalize_fc(obj_pair_feat4rel_rep)  # (num_rel, hidden_dim)

        return obj_pair_feat4rel_rep

    def forward(self, inst_roi_feats, union_features, inst_proposals, rel_pair_idxs):
        """

        :param inst_roi_feats: instance ROI features, list(Tensor)
        :param inst_proposals: instance proposals, list(BoxList())
        :param rel_pair_idxs:
        :return:
            obj_pred_logits obj_pred_labels 2nd time instance classification results
            obj_representation4rel, the objects features ready for the represent the relationship
        """
        # using label or logits do the label space embeddings
        if self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            obj_labels = cat([proposal.get_field("labels") for proposal in inst_proposals], dim=0)
        else:
            obj_labels = None

        if self.word_embed_feats_on:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:#obj_embed_by_pred_dist dim:200。obj_labels是把batch里的拼在一起
                obj_embed_by_pred_dist = self.obj_embed_on_prob_dist(obj_labels.long())#word embedding层，输入word标签得embedding
            else:
                obj_logits = cat([proposal.get_field("predict_logits") for proposal in inst_proposals], dim=0).detach()
                obj_embed_by_pred_dist = F.softmax(obj_logits, dim=1) @ self.obj_embed_on_prob_dist.weight

        # box positive geometry embedding
        assert inst_proposals[0].mode == 'xyxy'
        pos_embed = self.pos_embed(encode_box_info(inst_proposals))#这个文章用的position embedding

        # word embedding refine
        batch_size = inst_roi_feats.shape[0]

        if self.word_embed_feats_on:
            obj_pre_rep = cat((inst_roi_feats, obj_embed_by_pred_dist, pos_embed), -1)#4096,200,128.resentation:visual-word_embedding-位置编码
        else:
            obj_pre_rep = cat((inst_roi_feats, pos_embed), -1)
        # object level contextual feature
        augment_obj_feat = self.obj_hidden_linear(obj_pre_rep)  # map to hidden_dim 4424->512

        # todo reclassify on the fused object features
        # Decode in order
        if self.mode != 'predcls':
            # todo: currently no redo classification on embedding representation,
            #       we just use the first stage object prediction
            obj_pred_labels = cat([each_prop.get_field("pred_labels") for each_prop in inst_proposals], dim=0)
        else:
            assert obj_labels is not None
            obj_pred_labels = obj_labels

        # object labels space embedding from the prediction labels
        if self.word_embed_feats_on:
            obj_embed_by_pred_labels = self.obj_embed_on_pred_label(obj_pred_labels.long())

        if self.word_embed_feats_on:
            augment_obj_feat = cat((obj_embed_by_pred_labels, inst_roi_feats, augment_obj_feat), -1)#先前的augment_obj_feat是roi_feat+world_dis+pos_embe
        else:
            augment_obj_feat = cat((inst_roi_feats, augment_obj_feat), -1)

        if self.rel_feature_type == "obj_pair" or self.rel_feature_type == "fusion":
            rel_features = self.pairwise_rel_features(augment_obj_feat, union_features,#函数没有用union,完全是obj_feat运算，乘上了box的相关性
                                                      rel_pair_idxs, inst_proposals)
            if self.rel_feature_type == "fusion":
                if self.rel_feat_dim_not_match:
                    union_features = self.rel_feature_up_dim(union_features)
                rel_features = union_features + rel_features #可以认为fusion指的是把union特征以及单独得到的pair特征相加

        elif self.rel_feature_type == "union":
            if self.rel_feat_dim_not_match:
                union_features = self.rel_feature_up_dim(union_features)
            rel_features = union_features

        else:
            assert False
        # mapping to hidden
        augment_obj_feat = self.obj_feat_aug_finalize_fc(augment_obj_feat)
        # memorize average feature


        return augment_obj_feat, rel_features
