import numpy as np
import tensorflow as tf

from hparams.hparams import HParams
from custom_layers import TokenEmbedding, PositionalEmbedding
from utils.text.text_processing import create_look_ahead_mask
from custom_architectures.transformers_arch.transformer_arch import Transformer, HParamsTransformer

HParamsSpeechEmbedding  = HParams(
    num_layers  = 3,
    
    kernel_size = 11,
    use_bias    = True,
    strides     = 2,
    padding     = 'same',
    activation  = 'relu'
)

HParamsSpeechTransformer    = HParams(
    ** HParamsSpeechEmbedding.get_config(add_prefix = 'speech_embedding'),
    ** HParamsTransformer,
    
    vocab_size  = 32,
    
    sos_token_idx   = 0,
    eos_token_idx   = 1,
    
    max_input_length    = 4096,
    max_output_length   = 512,
)

class SpeechEmbedding(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, max_len, name = None, ** kwargs):
        super().__init__(name = name)
        self.hparams = HParamsSpeechEmbedding.extract(kwargs)
        self.hparams.embedding_dim  = embedding_dim
        self.hparams.max_len    = max_len
        
        self.convs = tf.keras.Sequential([
            tf.keras.layers.Conv1D(
                embedding_dim, kernel_size = self.hparams.kernel_size, strides = self.hparams.strides,
                use_bias = self.hparams.use_bias, padding = self.hparams.padding,
                activation = self.hparams.activation, name = 'conv_{}'.format(i + 1)
            ) for i in range(self.hparams.num_layers)
        ])

        
    def call(self, mels, training = False):
        embedded = self.convs(mels, training = training)
        return embedded
    
    def get_config(self):
        config = super().get_config()
        return self.hparams + config
        
class TransformerSTT(tf.keras.Model):
    def __init__(self, embedding_dim, vocab_size, input_shape, sos_token, eos_token, name = None, ** kwargs):
        super().__init__(name = name)
        self.hparams = HParamsSpeechTransformer.extract(kwargs)
        self.hparams.vocab_size = vocab_size
        self.hparams.sos_token  = sos_token
        self.hparams.eos_token  = eos_token
        self.hparams.embedding_dim  = embedding_dim
        self.hparams.input_shape    = input_shape
        self.hparams.n_mel_channels = input_shape[-1]
        
        self.sos_token  = sos_token
        self.eos_token  = eos_token
        
        self.mel_embedding  = SpeechEmbedding(
            embedding_dim = embedding_dim, max_len = self.hparams.max_input_length,
            ** self.hparams.get_config(prefix = 'speech_embedding'), name = 'mel_pos_embedding'
        )
        self.mel_pos_embedding = PositionalEmbedding(
            embedding_dim = embedding_dim, max_len = self.hparams.max_input_length,
            name = 'text_pos_embedding'
        )

        
        self.text_embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.text_pos_embedding = PositionalEmbedding(
            embedding_dim = embedding_dim, max_len = self.hparams.max_output_length,
            name = 'text_pos_embedding'
        )
        
        self.transformer = Transformer(** self.hparams)
        
        self.output_layer   = tf.keras.layers.Dense(
            vocab_size, activation = 'softmax', name = 'output_layer'
        )
        
    def _build(self):
        text_input  = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
        text_lengths = np.array([9])
        
        mel_input = np.random.normal(
            size = (1, 64, self.hparams.n_mel_channels)
        ).astype(np.float32)
        mel_lengths = np.array([64])
        
        inputs = [mel_input, mel_lengths, text_input, text_lengths]
        self(inputs, training = False)
        
    def call(self, inputs, training   = False, return_attention = True):
        mels, mel_lengths, text, text_lengths = inputs
        
        look_ahead_mask = create_look_ahead_mask(tf.shape(text)[0], tf.shape(text)[1])
        padding_mask    = tf.expand_dims(tf.sequence_mask(
            text_lengths, maxlen = tf.shape(text)[1], dtype = look_ahead_mask.dtype
        ), axis = -1)
        combined_mask   = tf.expand_dims(tf.maximum(look_ahead_mask, padding_mask), axis = 1)

        embedded_mels   = self.mel_embedding(mels, training = training)
        embedded_mels   = self.mel_pos_embedding(embedded_mels, training = training)
        
        embedded_text   = self.text_embedding(text, training = training)
        embedded_text   = self.text_pos_embedding(embedded_text, training = training)
        
        output, enc_attn_weights, dec_attn_weights = self.transformer(
            [embedded_mels, embedded_text], training = training, return_attention = True,
            encoder_padding_mask = None, look_ahead_mask = combined_mask
        )
        
        output = self.output_layer(output, training = training)
        
        return output if not return_attention else (output, dec_attn_weights)
    
    def infer(self, inputs, training   = False, return_attention = True, ** kwargs):
        mels, mel_lengths = inputs
        batch_size = tf.shape(mels)[0]
        
        embedded_mels   = self.mel_embedding(mels, training = training)
        embedded_mels   = self.mel_pos_embedding(embedded_mels, training = training)
        encoded_mels, enc_attn_weights  = self.transformer.encode(embedded_mels, training = training)
        
        
        tokens      = tf.zeros((batch_size, 1), dtype = tf.int32) + self.sos_token
        finished    = tf.zeros((batch_size, 1), dtype = tf.int32)
        
        logits      = tf.zeros((batch_size, 1, self.hparams.vocab_size))
        enc_attn_weights, dec_attn_weights = {}, {}
        
        while tf.shape(tokens)[1] < self.hparams.max_output_length and tf.reduce_sum(finished) < batch_size:
            look_ahead_mask = tf.expand_dims(
                create_look_ahead_mask(tf.shape(tokens)[0], tf.shape(tokens)[1]), axis = 1
            )
            
            embedded_text   = self.text_embedding(tokens, training = training)
            embedded_text   = self.text_pos_embedding(embedded_text, training = training)
            
            output, dec_attn_weights = self.transformer.decode(
                encoded_mels, embedded_text, training = training, look_ahead_mask = look_ahead_mask
            )
          
            logits = self.output_layer(output, training = training)
            
            
            new_tokens = tf.cast(tf.argmax(logits[:,-1:,:], axis = -1), tf.int32)
            
            finished = tf.maximum(finished, tf.cast(new_tokens == self.eos_token, tf.int32))
            tokens = tf.concat([tokens, new_tokens], axis = -1)
        
        return logits if not return_attention else (logits, dec_attn_weights)
    
    def get_config(self):
        return self.hparams.get_config()
    
    @classmethod
    def from_config(cls, config, custom_objects = None):
        return cls(** config)

custom_functions    = {
    'TransformerSTT'    : TransformerSTT,
    'transformer_stt'   : TransformerSTT
}

custom_objects  = {
    'Transformer'       : Transformer,
    'TokenEmbedding'    : TokenEmbedding,
    'SpeechEmbedding'   : SpeechEmbedding,
    'TransformerSTT'    : TransformerSTT
}