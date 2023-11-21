# import os
import tensorflow as tf
import logging






class PositionalEncoding(tf.keras.Model):
    """This class implements the absolute sinusoidal positional encoding function.

    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))

    Arguments
    ---------
    input_size: int
        Embedding dimension.
    max_len : int, optional
        Max length of the input sequences (default 2500).

    Example
    -------
    torch.Size([1, 120, 512])
    """

    def __init__(self, input_size, max_len=2500):
        super().__init__()
        logging.info("Initializing PositionalEncoding with input_size: {} and max_len: {}".format(input_size, max_len))
        self.max_len = max_len
        positions = tf.expand_dims(tf.range(0, self.max_len,dtype=float), axis=1)
        denominator = tf.exp(
            tf.range(0, input_size, 2,dtype=float)
            * -(tf.math.log(10000.0) / input_size)
        )
        pe = tf.concat([tf.stack([tf.sin(positions * denominator[i]), tf.cos(positions * denominator[i])],axis=1)
                   for i in range(denominator.shape[0])], axis=1)
        pe = tf.transpose(pe,(2, 0, 1))
        self.pe = pe

    def call(self, input, training=None, mask=None):
        """
        Arguments
        ---------
        x : tensor
            Input feature shape (batch, time, fea)
        """
        logging.info("PositionalEncoding call")
        return self.pe[:, : input.shape[1]]
    

class Transformer_Layer(tf.keras.Model):
    def __init__(self, num_heads=8, d_model=256, d_ffn=2048):
        super().__init__()
        logging.debug("Initializing Transformer Layer with num_heads: {}, d_model: {}, d_ffn: {}".format(num_heads, d_model, d_ffn))
        self.layer_norm1=tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2=tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads, key_dim=int(d_model/num_heads))
        self.pos_ffw1 = tf.keras.layers.Dense(d_ffn, activation='relu')
        self.pos_ffw2 = tf.keras.layers.Dense(d_model)


    def call(self, input, training=None, mask=None):
        logging.debug("Transformer Layer call")
        z1 = input
        # positional encoding
        z_perm = self.layer_norm1(z1)
        #z_perm = tf.transpose(z_perm, (1, 0, 2))
        
        # print('z_perm', z_perm.shape)
        z_perm = self.mha(z_perm, z_perm)
        
        #z_perm = tf.transpose(z_perm, (1, 0, 2))
        z2 = z_perm+z1
        z_perm = self.layer_norm2(z2)
        #z_perm = tf.transpose(z_perm,(1, 0, 2))
        z_perm = self.pos_ffw2(self.pos_ffw1(z_perm))
        #z_perm = tf.transpose(z_perm, (1, 0, 2))
        z3 = z_perm+z2
        return z3+z1

class Ca_Block(tf.keras.Model):
    block_list: list
    def __init__(self, num_heads=8, d_model=256, d_ffn=2048, k=4):
        super().__init__()
        logging.debug("Initializing Ca_Block with num_heads: {}, d_model: {}, d_ffn: {}, k: {}".format(num_heads, d_model, d_ffn, k))
        self.layer_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.block_list=[0]*k
        self.pos_enc = PositionalEncoding(d_model)  
        for i in range(k):
            self.block_list[i] = Transformer_Layer(num_heads=num_heads, d_model=d_model, d_ffn=d_ffn)

    def call(self, input, training=None, mask=None):
        logging.debug("Ca_Block call")
        z = input
        z1 = z + self.pos_enc(z)
        for i in range(len(self.block_list)):
            z1 = self.block_list[i](z1)
        z1 = self.layer_norm1(z1)
        return z1+z

class Sepformer(tf.keras.Model):
    def __init__(self, enc_filters=256, enc_filter_len=16, enc_filter_stride=8, chunk_len=250, chunk_overlap=0.5, speakers=2, k=8,num_layer=2):
        super().__init__()
        logging.info("Initializing Sepformer")
        self.enc_conv1=tf.keras.layers.Conv1D(enc_filters,enc_filter_len, enc_filter_stride, activation='relu')
        self.enc_group_norm=tf.keras.layers.GroupNormalization(groups=1, axis=-1,epsilon=1e-8)


        self.dec_conv1 = tf.keras.layers.Conv1DTranspose(1,enc_filter_len, enc_filter_stride)
        self.linear1 = tf.keras.layers.Dense(enc_filters)
        self.linear_speaker = tf.keras.layers.Dense(enc_filters*speakers)
        self.end_conv1x1 = tf.keras.layers.Dense(enc_filters, use_bias=False, activation='relu')
        self.final_output = tf.keras.layers.Dense(enc_filters, activation='tanh')
        self.final_gate = tf.keras.layers.Dense(enc_filters, activation='sigmoid')
        self.chunk_len = chunk_len
        self.chunk_overlap = chunk_overlap
        self.prelu = tf.keras.layers.PReLU(shared_axes=[1, 2])
        self.speakers=speakers
        self.inter_ca_list=[0]*num_layer
        self.intra_ca_list=[0]*num_layer
        self.intra_group_norm=[0]*num_layer
        self.inter_group_norm=[0]*num_layer
        self.num_layer=num_layer
        for i in range(num_layer):
            self.intra_group_norm[i] = tf.keras.layers.GroupNormalization(groups=1, axis=-1, epsilon=1e-8)
            self.inter_group_norm[i] = tf.keras.layers.GroupNormalization(groups=1, axis=-1, epsilon=1e-8)
            self.inter_ca_list[i] = Ca_Block(num_heads=8, d_model=enc_filters, d_ffn=1024, k=k)
            self.intra_ca_list[i] = Ca_Block(num_heads=8, d_model=enc_filters, d_ffn=1024, k=k)


    def call(self, input, training=None, mask=None):
        logging.info("Sepformer call")
        # encoder
        # x1 = tf.expand_dims(input, axis=2)
        # x1 = tf.expand_dims(input, axis=-1)  # Add a dimension of size 1  #FIXME: Add extra dim because of num_channels of the tensorflow

         # Ensure the input is expanded to 3D (batch_size, sequence_length, num_channels)
        # if len(input.shape) == 2:
        #     # Reshape input from shape (batch_size, sequence_length) to (batch_size, sequence_length, 1)
        #     input = tf.expand_dims(input, axis=-1) 


        x1 = self.enc_conv1(input)
        # preproc
        h = self.enc_group_norm(x1)
        h = self.linear1(h)

        # segmentation + padding
        overlap_size = int(self.chunk_len * self.chunk_overlap)
        calc = self.chunk_len - (tf.shape(h)[1] % self.chunk_len)
        pad = tf.zeros((tf.shape(h)[0], calc, tf.shape(h)[2]))  # Hd.shape[0] statt -1
        h1 = tf.concat((h, pad), axis=1)
        h2 = tf.signal.frame(h1, self.chunk_len, overlap_size, axis=-2)
        # Sepformer
        for i in range(self.num_layer):
            intra = tf.concat([h2[:,j,:,:] for j in range(h2.shape[1])], axis=0)
            intra = self.intra_ca_list[i](intra) #norm?
            intra = tf.stack(tf.split(intra, h2.shape[1], axis=0), axis=1)
            intra = self.intra_group_norm[i](intra)
            intra = intra+h2
            inter = tf.concat([intra[:,:,j,:] for j in range(intra.shape[2])], axis=0)
            inter = self.inter_ca_list[i](inter) #norm?
            inter = tf.stack(tf.split(inter, h2.shape[2], axis=0), axis=2)
            inter = self.inter_group_norm[i](inter)
            h2 = inter+intra

        # speaker split and overlap add
        h3 = self.prelu(h2)
        h3 = self.linear_speaker(h3)
        h3 = tf.concat(tf.split(h3,self.speakers, axis=-1),axis=0)
        h4 = tf.signal.overlap_and_add(tf.transpose(h3, (0,3,1,2) ), int(self.chunk_len * self.chunk_overlap))
        h4 = h4[:, :, :x1.shape[1]]
        h4 = tf.transpose(h4,(0,2,1))
        h4 = self.final_output(h4) * self.final_gate(h4)
        m = self.end_conv1x1(h4)
        m = tf.stack(tf.split(m, self.speakers, axis=0))

        # apply mask and decodeing
        x1 = tf.stack([x1] * self.speakers)
        sep_h = x1*m
        est_source = tf.concat([self.dec_conv1(sep_h[i]) for i in range(self.speakers)], axis=-1)
        est_source = tf.expand_dims(tf.transpose(est_source,(0,2,1)),-1)


        # print(f"** My Model est_source:{est_source} - {type(est_source)}") # FIXME: check the data type
        # print(f"** My Model est_source:{est_source[:][0][0][0][0]} - {type(est_source)}") # FIXME: check the data type
        
        return tf.cast(est_source, tf.float64)
