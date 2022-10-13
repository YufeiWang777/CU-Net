# ------------------------------------------------------------
# model for depth completion
# @author:                  jokerWRN
# @data:                    Mon 2021.1.22 16:53
# @latest modified data:    Mon 2020.1.22 16.53
# ------------------------------------------------------------
# ------------------------------------------------------------


from model.basic import *
from model.weights_init import *


class Model(nn.Module, ABC):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.geofeature = None
        self.geoplanes = 3
        if self.args.convolutional_layer_encoding == "xyz":
            self.geofeature = GeometryFeature()
        elif self.args.convolutional_layer_encoding == "std":
            self.geoplanes = 0
        elif self.args.convolutional_layer_encoding == "uv":
            self.geoplanes = 2
        elif self.args.convolutional_layer_encoding == "z":
            self.geoplanes = 1

        self.pooling = nn.AvgPool2d(kernel_size=2)
        self.sparsepooling = SparseDownSampleClose(stride=2)

        # Ben
        self.ben_conv_init = Convbnrelu(inplanes=1, planes=64, norm_layer=False, kernel_size=5, padding=2)

        self.ben_encoder_layer1_1 = BasicBlockGeo(inplanes=64, planes=64, stride=2, geoplanes=self.geoplanes)
        self.ben_encoder_layer1_2 = BasicBlockGeo(inplanes=64, planes=64, stride=1, geoplanes=self.geoplanes)
        self.ben_encoder_layer2_1 = BasicBlockGeo(inplanes=64, planes=64, stride=1, geoplanes=self.geoplanes)
        self.ben_encoder_layer2_2 = BasicBlockGeo(inplanes=64, planes=64, stride=1, geoplanes=self.geoplanes)

        self.ben_encoder_layer3_1 = BasicBlockGeo(inplanes=64, planes=128, stride=2, geoplanes=self.geoplanes)
        self.ben_encoder_layer3_2 = BasicBlockGeo(inplanes=128, planes=128, stride=1, geoplanes=self.geoplanes)
        self.ben_encoder_layer4_1 = BasicBlockGeo(inplanes=128, planes=128, stride=1, geoplanes=self.geoplanes)
        self.ben_encoder_layer4_2 = BasicBlockGeo(inplanes=128, planes=128, stride=1, geoplanes=self.geoplanes)

        self.ben_encoder_layer5_1 = BasicBlockGeo(inplanes=128, planes=256, stride=2, geoplanes=self.geoplanes)
        self.ben_encoder_layer5_2 = BasicBlockGeo(inplanes=256, planes=256, stride=1, geoplanes=self.geoplanes)
        self.ben_encoder_layer6_1 = BasicBlockGeo(inplanes=256, planes=256, stride=1, geoplanes=self.geoplanes)
        self.ben_encoder_layer6_2 = BasicBlockGeo(inplanes=256, planes=256, stride=1, geoplanes=self.geoplanes)

        self.ben_encoder_layer7_1 = BasicBlockGeo(inplanes=256, planes=256, stride=2, geoplanes=self.geoplanes)
        self.ben_encoder_layer7_2 = BasicBlockGeo(inplanes=256, planes=256, stride=1, geoplanes=self.geoplanes)
        self.ben_encoder_layer8_1 = BasicBlockGeo(inplanes=256, planes=256, stride=1, geoplanes=self.geoplanes)
        self.ben_encoder_layer8_2 = BasicBlockGeo(inplanes=256, planes=256, stride=1, geoplanes=self.geoplanes)

        self.ben_encoder_layer9_1 = BasicBlockGeo(inplanes=256, planes=256, stride=2, geoplanes=self.geoplanes)
        self.ben_encoder_layer9_2 = BasicBlockGeo(inplanes=256, planes=256, stride=1, geoplanes=self.geoplanes)
        self.ben_encoder_layer10_1 = BasicBlockGeo(inplanes=256, planes=256, stride=1, geoplanes=self.geoplanes)
        self.ben_encoder_layer10_2 = BasicBlockGeo(inplanes=256, planes=256, stride=1, geoplanes=self.geoplanes)

        # decoder
        self.ben_decoder_layer8 = Deconvbnrelu_pre(inplanes=256, planes=256, norm_layer=True, kernel_size=3, stride=2, padding=1,
                                           output_padding=1)
        self.ben_decoder_layer6 = Deconvbnrelu_pre(inplanes=256, planes=256, norm_layer=True, kernel_size=3, stride=2, padding=1,
                                           output_padding=1)
        self.ben_decoder_layer4 = Deconvbnrelu_pre(inplanes=256, planes=128, norm_layer=True, kernel_size=3, stride=2, padding=1,
                                           output_padding=1)
        self.ben_decoder_layer2 = Deconvbnrelu_pre(inplanes=128, planes=64, norm_layer=True, kernel_size=3, stride=2, padding=1,
                                           output_padding=1)
        self.ben_decoder_layer0 = Deconvbnrelu_pre(inplanes=64, planes=64, norm_layer=True, kernel_size=3, stride=2, padding=1,
                                           output_padding=1)

        self.ben_decoder_output_ref = BasicBlock(inplanes=64, planes=64, act=False)
        self.ben_predict_depth = conv1x1(inplanes=64, planes=1, stride=1, padding=1, bias=True)
        self.ben_predict_mask = nn.Sequential(
            conv1x1(inplanes=64, planes=1, stride=1, padding=1, bias=False),
            nn.Sigmoid()
        )

        # Jin
        self.jin_conv_init_dense = Convbnrelu(inplanes=1, planes=48, norm_layer=False, kernel_size=5, padding=2)
        self.jin_conv_init_sparse = Convbnrelu(inplanes=1, planes=16, norm_layer=False, kernel_size=5, padding=2)

        self.jin_encoder_layer1_1 = BasicBlockGeo(inplanes=64, planes=64, stride=2, geoplanes=self.geoplanes)
        self.jin_encoder_layer1_2 = BasicBlockGeo(inplanes=64, planes=64, stride=1, geoplanes=self.geoplanes)
        self.jin_encoder_layer2_1 = BasicBlockGeo(inplanes=64, planes=64, stride=1, geoplanes=self.geoplanes)
        self.jin_encoder_layer2_2 = BasicBlockGeo(inplanes=64, planes=64, stride=1, geoplanes=self.geoplanes)

        self.jin_encoder_layer3_1 = BasicBlockGeo(inplanes=64, planes=128, stride=2, geoplanes=self.geoplanes)
        self.jin_encoder_layer3_2 = BasicBlockGeo(inplanes=128, planes=128, stride=1, geoplanes=self.geoplanes)
        self.jin_encoder_layer4_1 = BasicBlockGeo(inplanes=128, planes=128, stride=1, geoplanes=self.geoplanes)
        self.jin_encoder_layer4_2 = BasicBlockGeo(inplanes=128, planes=128, stride=1, geoplanes=self.geoplanes)

        self.jin_encoder_layer5_1 = BasicBlockGeo(inplanes=128, planes=256, stride=2, geoplanes=self.geoplanes)
        self.jin_encoder_layer5_2 = BasicBlockGeo(inplanes=256, planes=256, stride=1, geoplanes=self.geoplanes)
        self.jin_encoder_layer6_1 = BasicBlockGeo(inplanes=256, planes=256, stride=1, geoplanes=self.geoplanes)
        self.jin_encoder_layer6_2 = BasicBlockGeo(inplanes=256, planes=256, stride=1, geoplanes=self.geoplanes)

        self.jin_encoder_layer7_1 = BasicBlockGeo(inplanes=256, planes=256, stride=2, geoplanes=self.geoplanes)
        self.jin_encoder_layer7_2 = BasicBlockGeo(inplanes=256, planes=256, stride=1, geoplanes=self.geoplanes)
        self.jin_encoder_layer8_1 = BasicBlockGeo(inplanes=256, planes=256, stride=1, geoplanes=self.geoplanes)
        self.jin_encoder_layer8_2 = BasicBlockGeo(inplanes=256, planes=256, stride=1, geoplanes=self.geoplanes)

        self.jin_encoder_layer9_1 = BasicBlockGeo(inplanes=256, planes=256, stride=2, geoplanes=self.geoplanes)
        self.jin_encoder_layer9_2 = BasicBlockGeo(inplanes=256, planes=256, stride=1, geoplanes=self.geoplanes)
        self.jin_encoder_layer10_1 = BasicBlockGeo(inplanes=256, planes=256, stride=1, geoplanes=self.geoplanes)
        self.jin_encoder_layer10_2 = BasicBlockGeo(inplanes=256, planes=256, stride=1, geoplanes=self.geoplanes)

        # decoder
        self.jin_decoder_layer8 = Deconvbnrelu_pre(inplanes=256, planes=256, norm_layer=True, kernel_size=3, stride=2, padding=1,
                                           output_padding=1)
        self.jin_decoder_layer6 = Deconvbnrelu_pre(inplanes=256, planes=256, norm_layer=True, kernel_size=3, stride=2, padding=1,
                                           output_padding=1)
        self.jin_decoder_layer4 = Deconvbnrelu_pre(inplanes=256, planes=128, norm_layer=True, kernel_size=3, stride=2, padding=1,
                                           output_padding=1)
        self.jin_decoder_layer2 = Deconvbnrelu_pre(inplanes=128, planes=64, norm_layer=True, kernel_size=3, stride=2, padding=1,
                                           output_padding=1)
        self.jin_decoder_layer0 = Deconvbnrelu_pre(inplanes=64, planes=64, norm_layer=True, kernel_size=3, stride=2, padding=1,
                                           output_padding=1)

        self.jin_decoder_output_ref = BasicBlock(inplanes=64, planes=64, act=False)
        self.jin_predict_depth = conv1x1(inplanes=64, planes=1, stride=1, padding=1, bias=True)
        self.jin_predict_mask = nn.Sequential(
            conv1x1(inplanes=64, planes=1, stride=1, padding=1, bias=False),
            nn.Sigmoid()
        )

        self.softmax = nn.Softmax(dim=1)

        if args.init == 'Gaussian_random':
            Gaussian_random(self)
        else:
            raise Exception('no init method is selected!')

        # Set parameter groups
        params = []
        for param in self.named_parameters():
            if param[1].requires_grad:
                params.append(param[1])

        params = nn.ParameterList(params)

        self.param_groups = [
            {'params': params, 'lr': self.args.lr}
        ]

        weights_7 = torch.tensor([[0, 0, 0, 1, 0, 0, 0],
                                [0, 0, 1, 1, 1, 0, 0],
                                [0, 1, 1, 1, 1, 1, 0],
                                [1, 1, 1, 1, 1, 1, 1],
                                [0, 1, 1, 1, 1, 1, 0],
                                [0, 0, 1, 1, 1, 0, 0],
                                [0, 0, 0, 1, 0, 0, 0],], dtype=torch.float32)
        weights_7 = weights_7.view(1, 1, 7, 7)
        self.myconv_7 = nn.Conv2d(1, 1, 7, padding=3, bias=False)
        self.myconv_7.weight = nn.Parameter(weights_7, requires_grad=False)

        weights_13 = torch.tensor([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                                   [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                   [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                   [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                                   [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                                   [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],], dtype=torch.float32)
        weights_13 = weights_13.view(1, 1, 13, 13)
        self.myconv_13 = nn.Conv2d(1, 1, 13, padding=6, bias=False)
        self.myconv_13.weight = nn.Parameter(weights_13, requires_grad=False)

    def forward(self, x):

        # =========================== GEOFEATURE ===============================
        position = x['position']
        unorm = position[:, 0:1, :, :]
        vnorm = position[:, 1:2, :, :]

        K = x['K']
        f352 = K[:, 1, 1].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        c352 = K[:, 1, 2].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        f1216 = K[:, 0, 0].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        c1216 = K[:, 0, 2].unsqueeze(1).unsqueeze(2).unsqueeze(3)

        vnorm_s2 = self.pooling(vnorm)
        vnorm_s3 = self.pooling(vnorm_s2)
        vnorm_s4 = self.pooling(vnorm_s3)
        vnorm_s5 = self.pooling(vnorm_s4)
        vnorm_s6 = self.pooling(vnorm_s5)

        unorm_s2 = self.pooling(unorm)
        unorm_s3 = self.pooling(unorm_s2)
        unorm_s4 = self.pooling(unorm_s3)
        unorm_s5 = self.pooling(unorm_s4)
        unorm_s6 = self.pooling(unorm_s5)

        d = x['dep']
        valid_mask = torch.where(d>0, torch.full_like(d, 1.0), torch.full_like(d, 0.0))
        d_s2, vm_s2 = self.sparsepooling(d, valid_mask)
        d_s3, vm_s3 = self.sparsepooling(d_s2, vm_s2)
        d_s4, vm_s4 = self.sparsepooling(d_s3, vm_s3)
        d_s5, vm_s5 = self.sparsepooling(d_s4, vm_s4)
        d_s6, vm_s6 = self.sparsepooling(d_s5, vm_s5)

        geo_s1 = None
        geo_s2 = None
        geo_s3 = None
        geo_s4 = None
        geo_s5 = None
        geo_s6 = None

        if self.args.convolutional_layer_encoding == "xyz":
            geo_s1 = self.geofeature(d, vnorm, unorm, 352, 1216, c352, c1216, f352, f1216)
            geo_s2 = self.geofeature(d_s2, vnorm_s2, unorm_s2, 352 / 2, 1216 / 2, c352, c1216, f352, f1216)
            geo_s3 = self.geofeature(d_s3, vnorm_s3, unorm_s3, 352 / 4, 1216 / 4, c352, c1216, f352, f1216)
            geo_s4 = self.geofeature(d_s4, vnorm_s4, unorm_s4, 352 / 8, 1216 / 8, c352, c1216, f352, f1216)
            geo_s5 = self.geofeature(d_s5, vnorm_s5, unorm_s5, 352 / 16, 1216 / 16, c352, c1216, f352, f1216)
            geo_s6 = self.geofeature(d_s6, vnorm_s6, unorm_s6, 352 / 32, 1216 / 32, c352, c1216, f352, f1216)
        elif self.args.convolutional_layer_encoding == "uv":
            geo_s1 = torch.cat((vnorm, unorm), dim=1)
            geo_s2 = torch.cat((vnorm_s2, unorm_s2), dim=1)
            geo_s3 = torch.cat((vnorm_s3, unorm_s3), dim=1)
            geo_s4 = torch.cat((vnorm_s4, unorm_s4), dim=1)
            geo_s5 = torch.cat((vnorm_s5, unorm_s5), dim=1)
            geo_s6 = torch.cat((vnorm_s6, unorm_s6), dim=1)
        elif self.args.convolutional_layer_encoding == "z":
            geo_s1 = d
            geo_s2 = d_s2
            geo_s3 = d_s3
            geo_s4 = d_s4
            geo_s5 = d_s5
            geo_s6 = d_s6

        # =========================== BEN BRANCH ===============================
        ben_feature0 = self.ben_conv_init(d)

        ben_feature1_1 = self.ben_encoder_layer1_1(ben_feature0, geo_s1, geo_s2)
        ben_feature1_2 = self.ben_encoder_layer1_2(ben_feature1_1, geo_s2, geo_s2)
        ben_feature2_1 = self.ben_encoder_layer2_1(ben_feature1_2, geo_s2, geo_s2)
        ben_feature2_2 = self.ben_encoder_layer2_2(ben_feature2_1, geo_s2, geo_s2)

        ben_feature3_1 = self.ben_encoder_layer3_1(ben_feature2_2, geo_s2, geo_s3)
        ben_feature3_2 = self.ben_encoder_layer3_2(ben_feature3_1, geo_s3, geo_s3)
        ben_feature4_1 = self.ben_encoder_layer4_1(ben_feature3_2, geo_s3, geo_s3)
        ben_feature4_2 = self.ben_encoder_layer4_2(ben_feature4_1, geo_s3, geo_s3)

        ben_feature5_1 = self.ben_encoder_layer5_1(ben_feature4_2, geo_s3, geo_s4)
        ben_feature5_2 = self.ben_encoder_layer5_2(ben_feature5_1, geo_s4, geo_s4)
        ben_feature6_1 = self.ben_encoder_layer6_1(ben_feature5_2, geo_s4, geo_s4)
        ben_feature6_2 = self.ben_encoder_layer6_2(ben_feature6_1, geo_s4, geo_s4)

        ben_feature7_1 = self.ben_encoder_layer7_1(ben_feature6_2, geo_s4, geo_s5)
        ben_feature7_2 = self.ben_encoder_layer7_2(ben_feature7_1, geo_s5, geo_s5)
        ben_feature8_1 = self.ben_encoder_layer8_1(ben_feature7_2, geo_s5, geo_s5)
        ben_feature8_2 = self.ben_encoder_layer8_2(ben_feature8_1, geo_s5, geo_s5)

        ben_feature9_1 = self.ben_encoder_layer9_1(ben_feature8_2, geo_s5, geo_s6)
        ben_feature9_2 = self.ben_encoder_layer9_2(ben_feature9_1, geo_s6, geo_s6)
        ben_feature10_1 = self.ben_encoder_layer10_1(ben_feature9_2, geo_s6, geo_s6)
        ben_feature10_2 = self.ben_encoder_layer10_2(ben_feature10_1, geo_s6, geo_s6)
            
        ben_feature_decoder8 = self.ben_decoder_layer8(ben_feature10_2, ben_feature8_2)
        ben_feature_decoder6 = self.ben_decoder_layer6(ben_feature_decoder8, ben_feature6_2)
        ben_feature_decoder4 = self.ben_decoder_layer4(ben_feature_decoder6, ben_feature4_2)
        ben_feature_decoder2 = self.ben_decoder_layer2(ben_feature_decoder4, ben_feature2_2)
        ben_feature_decoder0 = self.ben_decoder_layer0(ben_feature_decoder2, ben_feature0)
        
        ben_depth_ref = self.ben_decoder_output_ref(ben_feature_decoder0)
        ben_depth = self.ben_predict_depth(ben_depth_ref)
        ben_conf = self.ben_predict_mask(ben_depth_ref)

        # =========================== JIN BRANCH ===============================
        depth_mask_from_ben = ben_conf.detach()
        confidence_mask = torch.where(depth_mask_from_ben >= 0.7, torch.full_like(depth_mask_from_ben, 1.0), torch.full_like(depth_mask_from_ben, 0.0))
        confusion_mask = torch.where(depth_mask_from_ben < 0.7, torch.full_like(depth_mask_from_ben, 1.0), torch.full_like(depth_mask_from_ben, 0.0))
        d_confidece = d * confidence_mask

        valid_pixels = torch.where(d > 0.1, torch.full_like(d, 1.0), torch.full_like(d, 0.0))
        
        lidar_sum_7 = self.myconv_7(d)
        lidar_count_7 = self.myconv_7(valid_pixels)
        lidar_aveg_7 = lidar_sum_7 / (lidar_count_7 + 0.00001)
        potential_outliers_7 = torch.where((d - lidar_aveg_7) > 1.0, torch.full_like(d, 1.0), torch.full_like(d, 0.0))

        lidar_sum_13 = self.myconv_13(d)
        lidar_count_13 = self.myconv_13(valid_pixels)
        lidar_aveg_13 = lidar_sum_13 / (lidar_count_13 + 0.00001)
        potential_outliers_13 = torch.where((d - lidar_aveg_13) > 0.7, torch.full_like(d, 1.0), torch.full_like(d, 0.0))

        potential_outliers = potential_outliers_7 + potential_outliers_13
        lidar_clear = d * (1 - potential_outliers)
        d_confusion = lidar_clear * confusion_mask * valid_pixels
        d_clear = d_confidece + d_confusion

        jin_feature0_dense = self.jin_conv_init_dense(ben_depth)
        jin_feature0_sparse = self.jin_conv_init_sparse(d_clear)
        jin_feature0 = torch.cat((jin_feature0_dense, jin_feature0_sparse), dim=1)

        jin_feature1_1 = self.jin_encoder_layer1_1(jin_feature0, geo_s1, geo_s2)
        jin_feature1_2 = self.jin_encoder_layer1_2(jin_feature1_1, geo_s2, geo_s2)
        jin_feature2_1 = self.jin_encoder_layer2_1(jin_feature1_2, geo_s2, geo_s2)
        jin_feature2_2 = self.jin_encoder_layer2_2(jin_feature2_1, geo_s2, geo_s2)

        jin_feature3_1 = self.jin_encoder_layer3_1(jin_feature2_2, geo_s2, geo_s3)
        jin_feature3_2 = self.jin_encoder_layer3_2(jin_feature3_1, geo_s3, geo_s3)
        jin_feature4_1 = self.jin_encoder_layer4_1(jin_feature3_2, geo_s3, geo_s3)
        jin_feature4_2 = self.jin_encoder_layer4_2(jin_feature4_1, geo_s3, geo_s3)

        jin_feature5_1 = self.jin_encoder_layer5_1(jin_feature4_2, geo_s3, geo_s4)
        jin_feature5_2 = self.jin_encoder_layer5_2(jin_feature5_1, geo_s4, geo_s4)
        jin_feature6_1 = self.jin_encoder_layer6_1(jin_feature5_2, geo_s4, geo_s4)
        jin_feature6_2 = self.jin_encoder_layer6_2(jin_feature6_1, geo_s4, geo_s4)

        jin_feature7_1 = self.jin_encoder_layer7_1(jin_feature6_2, geo_s4, geo_s5)
        jin_feature7_2 = self.jin_encoder_layer7_2(jin_feature7_1, geo_s5, geo_s5)
        jin_feature8_1 = self.jin_encoder_layer8_1(jin_feature7_2, geo_s5, geo_s5)
        jin_feature8_2 = self.jin_encoder_layer8_2(jin_feature8_1, geo_s5, geo_s5)

        jin_feature9_1 = self.jin_encoder_layer9_1(jin_feature8_2, geo_s5, geo_s6)
        jin_feature9_2 = self.jin_encoder_layer9_2(jin_feature9_1, geo_s6, geo_s6)
        jin_feature10_1 = self.jin_encoder_layer10_1(jin_feature9_2, geo_s6, geo_s6)
        jin_feature10_2 = self.jin_encoder_layer10_2(jin_feature10_1, geo_s6, geo_s6)

        jin_feature_decoder8 = self.jin_decoder_layer8(jin_feature10_2, jin_feature8_2)
        jin_feature_decoder6 = self.jin_decoder_layer6(jin_feature_decoder8, jin_feature6_2)
        jin_feature_decoder4 = self.jin_decoder_layer4(jin_feature_decoder6, jin_feature4_2)
        jin_feature_decoder2 = self.jin_decoder_layer2(jin_feature_decoder4, jin_feature2_2)
        jin_feature_decoder0 = self.jin_decoder_layer0(jin_feature_decoder2, jin_feature0)

        jin_depth_ref = self.jin_decoder_output_ref(jin_feature_decoder0)
        jin_depth = self.jin_predict_depth(jin_depth_ref)
        jin_conf = self.jin_predict_mask(jin_depth_ref)

        ben_conf, jin_conf = torch.chunk(self.softmax(torch.cat((ben_conf, jin_conf), dim=1)), 2, dim=1)
        an_depth = ben_conf*ben_depth + jin_conf*jin_depth

        output = {'an_depth': an_depth, 'ben_depth':ben_depth, 'jin_depth':jin_depth,
                  'ben_mask':d_clear, 'ben_conf':ben_conf, 'jin_conf':jin_conf}

        return output

