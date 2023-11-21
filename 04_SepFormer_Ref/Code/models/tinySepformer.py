import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import yaml
import tensorflow as tf

# Path to source folder
path_to_src = '/isip/Users/spruenken/Documents/Thesis/src'

# Load parameters from configs file
with open(os.path.join(path_to_src, "tinySepformer_config.yaml"), "r") as f:
    config = yaml.safe_load(f)


# Methods that are built in as layers
def chunk_tensor(Hd):
    """ Layer of Chunk which windows the encoded audio tensor
        :param Hd: encoded audio tensor (EagerTensor) with TfxD (Tf:= folded time series, D := number of channels)
        :return: EagerTensor with TsxSxD (Ts := number of chunk vectors, S := size of chunk vector, D := number of channels)
    """
    # Get configuration
    chunk_size = config['chunk']['S']
    overlap_factor = config['chunk']['overlap']
    pad_end = config['chunk']['padEnd']

    # Calculate overlap size
    overlap_size = int(chunk_size * overlap_factor)

    # Padding the time series so that vector size is a multiple of chunk size
    if pad_end:
        calc = chunk_size - (tf.shape(Hd)[1] % chunk_size)
        pad = tf.zeros((tf.shape(Hd)[0], calc, tf.shape(Hd)[2]))
        Hd = tf.concat((Hd, pad), axis=1)

    # Create windowing
    Hs = tf.signal.frame(Hd, chunk_size, overlap_size, axis=-2)

    return Hs


def overlapAdd_tensor(Hdk):
    """ Layer of OverlapAdd which merges the chunked blocks
        :param Hdk: calculated mask tensor (EagerTensor) with TsxSx(DxK) (Ts := number of chunk vectors, S := size of chunk vector, D := number of channels, K := number of speakers)
        :return: Ho: overlaped mask tensor (EagerTensor) with TxDxK (T := length of time series, D := number of channels,  K := number of speakers)
    """
    # Get configuration
    frame_size = config['overlapAdd']['frameSize']
    overlap_factor = config['overlapAdd']['overlap']
    pad_end = config['chunk']['padEnd']
    pad_size = int((config['general']['wavvectorSize'] - config['encoder']['kernelSize']) / config['encoder']['strideFactor'] + 1)

    # Calculate overlap size
    overlap_size = int(frame_size * overlap_factor)

    # Creating overlap-add
    Hdk_trans = tf.transpose(Hdk, perm=[0, 3, 1, 2])
    Ho_trans = tf.signal.overlap_and_add(Hdk_trans, overlap_size)
    Ho = tf.transpose(Ho_trans, perm=[0, 2, 1])

    # In case of padding, remove the added values
    if pad_end:
        Ho = tf.slice(Ho, [0, 0, 0], (-1, pad_size, Ho.shape[2]))

    return Ho


def positional_encoding(Hs):
    """ Encodes at the beginning of the CA block the position to the input
    :param Hs: input feature of the CA-Block (EagerTensor) with TsxSxD (Ts := number of chunk vectors, S := size of chunk vector, D := number of channels)
    :return: Hs: same tensor (EagerTensor) with added position encoding
    """
    sequence_length, depth = tf.shape(Hs)[1], tf.shape(Hs)[2]
    Hs = tf.cast(Hs, dtype=tf.float64)

    # Calculate sine and cosine as a general position vector
    depth = depth / 2  # 2i/d = i/(d/2)

    positions = tf.cast(tf.expand_dims(tf.range(sequence_length), axis=1), dtype=tf.float64)
    depths = tf.cast(tf.expand_dims(tf.range(depth), axis=0) / depth, dtype=tf.float64)

    angle_rates = 1 / tf.math.pow(tf.cast(10000, dtype=tf.float64), depths)
    angle_rads = positions * angle_rates

    pos_encoding = tf.concat([tf.sin(angle_rads), tf.cos(angle_rads)], axis=-1)
    pos_encoding = tf.expand_dims(pos_encoding, axis=0)
    pos_encoding = tf.tile(pos_encoding, [tf.shape(Hs)[0], 1, 1])

    # Apply position encoding vector to input
    Hs = tf.add(Hs, pos_encoding)

    return Hs


# Components of the CA-Blocks
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, key_dim):
        super().__init__()
        # Parameters
        self.num_heads = config['caBlock']['numHeads']
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=key_dim)
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, Has):
        Ha = self.mha(query=Has, value=Has, key=Has)
        Ha = self.add([Has, Ha])
        Ha = self.layer_norm(Ha)
        return Ha


class SeparableConvolution(tf.keras.layers.Layer):
    def __init__(self, kernel_size, filters):
        super().__init__()
        self.separable_conv = tf.keras.layers.SeparableConv1D(kernel_size=kernel_size, filters=filters,
                                                              data_format='channels_last', padding='same')
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, Hcs):
        Hpc = self.separable_conv(Hcs)
        Hc = self.add([Hcs, Hpc])
        Hc = self.layer_norm(Hc)
        return Hc


class FeedForward(tf.keras.layers.Layer):
    def __init__(self, caType):
        super().__init__()
        # Parameters
        self.Df = config[caType]['feedForward']['Df']
        self.d_model = config[caType]['feedForward']['filtersModel']
        #self.dropout_rate = config[caType]['feedForward']['dropoutRate']  # Hidden, because this layer was not implemented in the Sepformer
        # Layers
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Dense(self.Df, activation='relu'),
            tf.keras.layers.Dense(self.d_model),
            # tf.keras.layers.Dropout(self.dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, HaHc):
        HaHc = self.add([HaHc, self.seq(HaHc)])
        HaHc = self.layer_norm(HaHc)
        return HaHc


class CABlock(tf.keras.layers.Layer):
    def __init__(self, caType):
        super().__init__()
        # Parameters
        self.Da = config[caType]['general']['Da']
        self.Dc = config[caType]['general']['Dc']
        self.num_heads = config['caBlock']['numHeads']
        self.key_dims = int(self.Da / self.num_heads)
        self.kernel_size = config[caType]['separableConvolution']['kernelSize']
        # Layers
        self.layer_norm_after_posenc = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.positional_encoding = tf.keras.layers.Lambda(positional_encoding)
        self.multihead_attention = MultiHeadAttention(key_dim=self.key_dims)
        self.conv_attention = SeparableConvolution(kernel_size=self.kernel_size, filters=self.Dc)
        self.add = tf.keras.layers.Add()
        self.layer_norm_after_mhaconv = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.feed_forward = FeedForward(caType=caType)
        self.layer_norm_after_ca_block = tf.keras.layers.LayerNormalization(epsilon=1e-8)

    def call(self, Hs):
        Hs_shape = Hs
        Hs = tf.concat([Hs[:,j,:,:] for j in range(Hs_shape.shape[1])], axis=0)
        Hs = self.positional_encoding(Hs)
        Hs_inner = self.layer_norm_after_posenc(Hs)
        Ha = self.multihead_attention(Hs_inner[:, :, 0:self.Da])
        Hc = self.conv_attention(Hs_inner[:, :, -self.Dc:])
        Hca = tf.concat([Ha, Hc], axis=-1)
        Hca_skip = self.add([Hca, Hs])
        Hca = self.layer_norm_after_mhaconv(Hca_skip)
        Hca = self.feed_forward(Hca)
        Hca = self.add([Hca, Hca_skip])
        Hca = tf.stack(tf.split(Hca, Hs_shape.shape[1], axis=0), axis=1)
        Hca = self.layer_norm_after_ca_block(Hca)
        return Hca


class OneMask(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        # Parameters
        self.n_intra = config['intraCA']['general']['nTimes']
        self.n_inter = config['interCA']['general']['nTimes']
        self.para_sharing = config['caBlock']['parameterSharing']
        # Layers
        self.layer_norm_after_ca_list = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.add = tf.keras.layers.Add()
        if not self.para_sharing:
            self.intra_list = []
            self.inter_list = []
            for i in range(self.n_intra):
                self.intra_list.append(CABlock('intraCA'))
            for i in range(self.n_inter):
                self.inter_list.append(CABlock('interCA'))
        else:
            self.intra_ca = CABlock('intraCA')
            self.inter_ca = CABlock('interCA')
        self.permute = tf.keras.layers.Permute((2, 1, 3))

    def call(self, Hca):
        Hca_input_intra = Hca
        if self.para_sharing:
            for n_intra in range(self.n_intra):
                Hca_input_intra = self.intra_ca(Hca_input_intra)
            Hca_input_intra = self.layer_norm_after_ca_list(Hca_input_intra)
            Hca_input_intra = self.add([Hca_input_intra, Hca])
            Hca_input_inter = self.permute(Hca_input_intra)
            for n_inter in range(self.n_inter):
                Hca_input_inter = self.inter_ca(Hca_input_inter)
            Hca_input_inter = self.permute(Hca_input_inter)
            Hca_input_inter = self.layer_norm_after_ca_list(Hca_input_inter)
            Hca = self.add([Hca_input_intra, Hca_input_inter])
        else:
            for intra_block in self.intra_list:
                Hca_input_intra = intra_block(Hca_input_intra)
            Hca_input_intra = self.layer_norm_after_ca_list(Hca_input_intra)
            Hca_input_intra = self.add([Hca_input_intra, Hca])
            Hca_input_inter = self.permute(Hca_input_intra)
            for inter_block in self.inter_list:
                Hca_input_inter = inter_block(Hca_input_inter)
            Hca_input_inter = self.permute(Hca_input_inter)
            Hca_input_inter = self.layer_norm_after_ca_list(Hca_input_inter)
            Hca = self.add([Hca_input_intra, Hca_input_inter])
        return Hca


# Inner components of the Masking Network
class Chunk(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        # Parameters
        self.parameter_dense = config['chunk']['filters']
        # Layers
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-8)
        self.dense = tf.keras.layers.Dense(self.parameter_dense)
        self.chunk = tf.keras.layers.Lambda(chunk_tensor)

    def call(self, H):
        H = self.layer_norm(H)
        Hd = self.dense(H)
        Hs = self.chunk(Hd)
        return Hs


class NMask(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        # Parameters
        self.nmask = config['nMask']['nTimes']
        # Layers
        self.one_mask_list = []
        for i in range(self.nmask):
            self.one_mask_list.append(OneMask())

    def call(self, Hca):
        for nMask in self.one_mask_list:
            Hca = nMask(Hca)
        return Hca


class OverlapAdd(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        # Parameters
        self.frame_size = config['overlapAdd']['frameSize']
        self.K = config['overlapAdd']['k']
        self.D = config['overlapAdd']['d']
        # Layer
        self.dense = tf.keras.layers.Dense(units=self.D * self.K)
        self.prelu = tf.keras.layers.PReLU(shared_axes=[1, 2])
        self.overlap_add = tf.keras.layers.Lambda(overlapAdd_tensor)

    def call(self, Hca):
        Hdk = self.dense(Hca)
        Hdk = self.prelu(Hdk)
        Hdk = tf.concat(tf.split(Hdk, self.K, axis=-1),axis=0)
        Ho = self.overlap_add(Hdk)
        return Ho


class Split(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        # Parameters
        self.filters = config['split']['filters']
        self.k = config['split']['speaker']
        # Layers
        self.final_output = tf.keras.layers.Dense(self.filters, activation='tanh')
        self.final_gate = tf.keras.layers.Dense(self.filters, activation='sigmoid')
        self.end_conv1x1 = tf.keras.layers.Dense(self.filters, use_bias=False, activation='relu')

    def call(self, Ho):
        print('Dimension Ho', Ho.shape)
        Ho_dense = tf.math.multiply(self.final_output(Ho), self.final_gate(Ho))
        M = self.end_conv1x1(Ho_dense)
        print('Dimension M', M.shape)
        M = tf.stack(tf.split(M, self.k, axis=0))
        return M




# Three main components
class Encoder(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.conv1d = tf.keras.layers.Conv1D(filters=config['encoder']['filters'],
                                             kernel_size=config['encoder']['kernelSize'],
                                             strides=config['encoder']['strideFactor'], activation='relu')

    def call(self, X):
        H = self.conv1d(X)
        return H


class MaskingNetwork(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.t_shape = int((config['general']['wavvectorSize'] - config['encoder']['kernelSize']) / config['encoder']['strideFactor'] + 1)
        self.d_shape = config['maskingNetwork']['filters']
        self.k = config['maskingNetwork']['numberSpeakers']
        self.chunk = Chunk()
        self.nmask = NMask()
        self.overlap_add = OverlapAdd()
        self.split = Split()
        self.dense_one_channel = tf.keras.layers.Dense(units=512, activation='relu')
        self.dense_two_time = tf.keras.layers.Dense(units=999, activation='relu')

    def call(self, H):
        Hs = self.chunk(H)
        Hca = self.nmask(Hs)
        Ho = self.overlap_add(Hca)
        M = self.split(Ho)
        return M


class Decoder(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        # Parameters
        self.k = config['general']['numberSpeaker']
        # Layers
        self.conv1d_trans = tf.keras.layers.Conv1DTranspose(filters=config['decoder']['filters'],
                                                            kernel_size=config['decoder']['kernelSize'],
                                                            strides=config['decoder']['strideFactor'])

    def call(self, HM):
        H, M = HM
        H = tf.stack([H] * self.k)
        X_roof_pre = tf.math.multiply(H, M)
        X_roof = tf.concat([self.conv1d_trans(X_roof_pre[i]) for i in range(self.k)], axis=-1)
        X_roof = tf.expand_dims(tf.transpose(X_roof,(0,2,1)),-1)
        X_roof = tf.cast(X_roof, dtype=tf.float64)
        return X_roof


# Tiny-Sepformer Model
class TinySepformer(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.masking_network = MaskingNetwork()
        self.decoder = Decoder()
#
    def call(self, X, training=None, mask=None):
        H = self.encoder(X)
        M = self.masking_network(H)
        X_roof = self.decoder([H, M])
        return X_roof
