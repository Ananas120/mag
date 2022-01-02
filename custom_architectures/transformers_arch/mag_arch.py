import json
import tensorflow as tf

from tensorflow.keras.models import model_from_json

from hparams.hparams import HParams
from utils.text import create_padding_mask
from custom_layers import FasterEmbedding
from custom_architectures.transformers_arch.transformer_arch import format_output
from custom_architectures.transformers_arch.bart_arch import BartEncoder, BartDecoder, Bart, HParamsBartEncoder, HParamsBart, _shared_keys


HParamsMAGEncoder = HParamsBartEncoder(
    repeat_pos_idx      = False,

    subsample_at    = -1,
    subsample_after = True,
    
    subsampling_step    = -1,
    subsampling_offset  = 1,
    subsampling_mode    = 'select',
    subsampling_drop_rate   = 0.,
    
    use_type_embedding      = False,
    random_training_type    = True,
    max_types   = 16
)

HParamsMAG  = HParamsBart(
    ** HParamsMAGEncoder.get_config(add_prefix = 'encoder')
)

class MAGEncoder(BartEncoder):
    def __init__(self, vocab_size, embedding_dim, name = None, ** kwargs):
        super().__init__(vocab_size = vocab_size, embedding_dim = embedding_dim, name = name, ** kwargs)

        self.hparams = HParamsMAGEncoder.extract(kwargs)
        self.hparams = self.hparams(vocab_size = vocab_size, embedding_dim = embedding_dim)
        
        self.type_embedding_layer = None
        if self.hparams.use_type_embedding:
            self.type_embedding_layer = FasterEmbedding(
                self.hparams.max_types, self.embedding_dim, name = "type_embedding"
            )
    
    @property
    def N(self):
        layer_idx = self.hparams.subsample_at
        if layer_idx < 0: layer_idx = len(self.encoder_layers) + layer_idx
        if self.hparams.subsample_after: layer_idx += 1
        return max(0, min(len(self.encoder_layers), layer_idx))
    
    @property
    def embedding_layers(self):
        return self.encoder_layers[: self.N]
    
    @property
    def memory_layers(self):
        return self.encoder_layers[self.N :]
    
    def _build(self):
        batch_size, q_seq_len, c_seq_len = 2, 16, 32
        
        q_tokens    = tf.ones([batch_size, q_seq_len], dtype = tf.int32)
        q_length    = tf.fill([batch_size, 1], q_seq_len)
        
        c_tokens    = tf.ones([batch_size, c_seq_len], dtype = tf.int32)
        c_length    = tf.fill([batch_size, 1], c_seq_len)
        
        self([q_tokens, q_length, c_tokens, c_length], training = False)
        self._maybe_init_subsampling_layer()
    
    def concat(self, embeddings, mask = None, training = False, merge_contexts = False, debug = False,
               ** kwargs):
        question, contexts = embeddings[0], embeddings[1:]
        q_mask, c_masks     = (mask[0], mask[1:]) if mask is not None else (None, None)
        
        c_lengths   = [tf.shape(c)[1] for c in contexts]
        contexts    = tf.concat(contexts, axis = 1) if len(contexts) > 1 else contexts[0]
        if c_masks is not None: c_masks = tf.concat(c_masks, axis = -1) if len(c_masks) > 1 else c_masks[0]
        
        lengths     = [tf.shape(question)[1]] + c_lengths
        
        if debug:
            tf.print("Sequence lengths :", lengths)
            tf.print("Question shape :", tf.shape(question))
            tf.print("Contexts shape :", tf.shape(contexts))
            tf.print("Masks shape :", tf.shape(c_masks) if c_masks is not None else None)
        
        if len(tf.shape(contexts)) == 4:
            if len(c_lengths) > 1:
                raise NotImplementedError("When passing multiple document / batch at once you cannot pass multiple contexts, please flatten everything !")
            if merge_contexts:
                raise NotImplementedError("You cannot pass multiple documents / batch and merge them !")
            
            types = tf.concat([
                tf.fill([tf.shape(question)[1]], 0),
                tf.repeat(tf.range(1, tf.shape(contexts)[1] + 1), tf.shape(contexts)[2])
            ], axis = -1)
            
            contexts    = tf.reshape(contexts, [tf.shape(contexts)[0], -1, tf.shape(contexts)[-1]])
            c_masks     = tf.reshape(c_masks, [tf.shape(c_masks)[0], 1, 1, -1])

            if debug:
                tf.print("Contexts (after flattening) shape :", tf.shape(contexts))
                tf.print("Masks (after flattening) shape :", tf.shape(c_masks))
            
        elif merge_contexts and tf.shape(question)[0] > 1:
            if len(c_lengths) > 1:
                raise NotImplementedError("When merging contexts, you can only pass 1 context / batch !")
            
            contexts = tf.reshape(
                tf.tile(contexts, [tf.shape(contexts)[0], 1, 1]), 
                [tf.shape(contexts)[0], -1, tf.shape(contexts)[-1]]
            )
            c_masks = tf.reshape(
                tf.tile(c_masks, [tf.shape(c_masks)[0], 1, 1, 1]), 
                [tf.shape(c_masks)[0], 1, 1, -1]
            )
            
            if debug:
                tf.print("Contexts (after merging) shape :", tf.shape(contexts))
                tf.print("Masks (after merging) shape :", tf.shape(c_masks))
            
            types = tf.concat([
                tf.fill([tf.shape(question)[1]], 0),
                tf.repeat(tf.range(1, tf.shape(question)[0] + 1), c_lengths[0])
            ], axis = -1)
        else:
            types   = tf.concat([
                tf.fill([length], i) for i, length in enumerate(lengths)
            ], axis = -1)
        
        memory  = tf.concat([question, contexts], axis = 1)
        masks   = tf.concat([q_mask, c_masks], axis = -1) #if q_mask is not None else None
        types   = tf.tile(tf.expand_dims(types, axis = 0), [tf.shape(question)[0], 1])

        return (memory, masks, types)
    
    def embed_types(self, memory, types, training = False, debug = False, ** kwargs):
        if self.type_embedding_layer is None: return memory, types
        
        if self.hparams.max_types == 2:
            types = tf.cast(types > 0, tf.int32)
        elif self.hparams.random_training_type and training and tf.reduce_max(types) < self.hparams.max_types:
            random_offset = tf.random.uniform(
                (tf.shape(types)[0], 1),
                minval  = 0,
                maxval  = self.hparams.max_types - tf.reduce_max(types),
                dtype   = tf.int32
            )
            types = types + (random_offset * tf.cast(types > 0, tf.int32))
        
        if debug:
            tf.print("Types used :", types)
        
        memory = memory + self.type_embedding_layer(types)
        
        return memory, types
    
    def embed(self, text, text_lengths = None, mask = None, training = False, positional_offset = -1,
              force_not_subsampling = False, debug = False, ** kwargs):
        if isinstance(text, (list, tuple)):
            assert len(text) % 2 == 0

            if len(text) > 2:
                if debug: tf.print("Force not subsampling :", force_not_subsampling)
                if not isinstance(force_not_subsampling, (list, tuple)):
                    force_not_subsampling = [force_not_subsampling] * (len(text) // 2)
                assert len(force_not_subsampling) == len(text) // 2
                
                embeddings, attn, states, masks = [], [], [], []
                for i in range(0, len(text), 2):
                    embedding_i, attn_i, states_i, mask_i = self.embed(
                        text[i], text[i+1], training = training, positional_offset = positional_offset,
                        force_not_subsampling = force_not_subsampling[i // 2], debug = debug
                    )
                    embeddings.append(embedding_i)
                    attn.append(attn_i)
                    states.append(states_i)
                    masks.append(mask_i)
                
                return embeddings, attn, states, masks
            
            text, text_lengths = text
        
        if debug:
            tf.print("Input tokens shape :", tf.shape(text), "-", tf.shape(text_lengths))
        
        attn_outputs, states_outputs = {}, {}
        
        batch_size = tf.shape(text)[0]
        n_doc_per_batch = -1
        if len(tf.shape(text)) == 3:
            n_doc_per_batch = tf.shape(text)[1]
            text            = tf.reshape(text, [-1, tf.shape(text)[-1]])
            text_lengths    = tf.reshape(text_lengths, [-1])
            if debug:
                tf.print("Input tokens reshaped shape :", tf.shape(text))
        
        if mask is None:
            mask = create_padding_mask(text, seq_len = text_lengths)

        embedded = self.embed_tokens(
            text, training = training, positional_offset = positional_offset,
            repeat_position = -1 if force_not_subsampling or not self.hparams.repeat_pos_idx else self.hparams.subsampling_step
        )
        
        output = embedded
        for i, layer in enumerate(self.embedding_layers):
            output, attn_weights = layer(
                output, mask = mask, training = training, return_attention = True
            )
            attn_outputs['emb_attn_{}'.format(layer.name)] = attn_weights
            states_outputs['emb_attn_{}'.format(layer.name)] = output
        
        if not force_not_subsampling:
            output, mask = self.subsample(output, mask = mask, training = training)
        
        if n_doc_per_batch != -1:
            output  = tf.reshape(output, [batch_size, n_doc_per_batch, tf.shape(output)[1], tf.shape(output)[-1]])
            mask    = tf.reshape(mask,   [batch_size, n_doc_per_batch, 1, 1, tf.shape(mask)[-1]])
        
        if debug:
            tf.print("Output subsampled shape :", tf.shape(output))
        
        return output, attn_outputs, states_outputs, mask
    
    def process_memory(self, embeddings, mask = None, training = False, ** kwargs):
        attn_outputs, states_outputs = {}, {}

        memory, mask, types = self.concat(embeddings, mask = mask, training = training, ** kwargs)
        
        memory, types = self.embed_types(memory, types, training = training, ** kwargs)
        
        output = memory
        for i, layer in enumerate(self.memory_layers):
            output, attn_weights = layer(
                output, mask = mask, training = training, return_attention = True
            )
            attn_outputs['memory_attn_{}'.format(layer.name)] = attn_weights
            states_outputs['emb_attn_{}'.format(layer.name)] = output

        return output, attn_outputs, states_outputs, types, mask
    
    def call(self,
             inputs,
             mask       = None,
             training   = False,
             
             merge_contexts     = False,
             positional_offset  = -1, 
             
             return_attention   = None,
             return_states  = None,
             return_mask    = None,
             
             ** kwargs
            ):
        embeddings, attn, states, masks = self.embed(
            inputs, mask = mask, training = training, positional_offset = positional_offset, ** kwargs
        )
        
        output, memory_attn, memory_states, types, memory_mask = self.process_memory(
            embeddings, mask = masks, training = training, merge_contexts = merge_contexts, ** kwargs
        )
        
        out = self.encoder.format_output(
            output, attn_weights = attn + [memory_attn], mask = memory_mask, states = states + [memory_states],
            return_attention = return_attention, return_mask = return_mask, return_states = return_states
        )
        return (out, types) if not isinstance(out, tuple) else out + (types, )
    
class MAG(Bart):
    def __init__(self, vocab_size, embedding_dim, max_input_length,
                 sos_token = None, eos_token = None, name = None, ** kwargs):
        super(Bart, self).__init__(name = name)
        
        tokens = {'sos_token' : sos_token, 'eos_token' : eos_token}
        if sos_token is not None:
            tokens.update({'decoder_sos_token' : sos_token, 'decoder_eos_token' : eos_token})
        
        kwargs.update({
            'embedding_dim' : embedding_dim, 'vocab_size' : vocab_size, 'max_input_length' : max_input_length
        })
        self.hparams = HParamsMAG.extract(kwargs)
        self.hparams = self.hparams(
            encoder_name    = 'encoder',
            decoder_name    = 'decoder',
            ** {'encoder_{}'.format(k) : self.hparams[k] for k in _shared_keys},
            ** {'decoder_{}'.format(k) : self.hparams[k] for k in _shared_keys},
            ** tokens
        )
        
        self.shared_embedding = FasterEmbedding(vocab_size, embedding_dim, name = "token_embedding")
        
        self.encoder    = MAGEncoder(
            token_embedding = self.shared_embedding,
            ** self.hparams.get_config(prefix = 'encoder')
        )
        self.decoder    = BartDecoder(
            token_embedding = self.shared_embedding,
            ** self.hparams.get_config(prefix = 'decoder')
        )
    
    def _build(self):
        batch_size, q_in_seq_len, c_in_seq_len, out_seq_len = 2, 16, 32, 8
        
        q_tokens    = tf.ones([batch_size, q_in_seq_len], dtype = tf.int32)
        q_length    = tf.fill([batch_size, 1], q_in_seq_len)
        
        c_tokens    = tf.ones([batch_size, c_in_seq_len], dtype = tf.int32)
        c_length    = tf.fill([batch_size, 1], c_in_seq_len)
        
        text = tf.ones([batch_size, out_seq_len], dtype = tf.int32)
        text_length = tf.fill([batch_size, 1], out_seq_len)
        
        self([q_tokens, q_length, c_tokens, c_length, text, text_length], training = False)
        self.encoder._maybe_init_subsampling_layer()

    def encode(self, inputs, mask = None, training = False, ** kwargs):
        return self.encoder(inputs, mask = mask, training = training, ** kwargs)
    
    def decode(self,
               encoder_out,
               decoder_inputs   = None,
               encoder_out_types    = None,
               
               decoder_mask = None,
               enc_padding_mask = None,
               
               training = False,
               
               ** kwargs
              ):
        if decoder_inputs is not None:
            output = self.decoder(
                [encoder_out, decoder_inputs[0], decoder_inputs[1]],
                mask    = decoder_mask,
                training    = training,
                enc_padding_mask    = enc_padding_mask,
                ** kwargs
            )
        else:
            output = self.decoder.infer(
                encoder_out,
                training    = training,
                enc_padding_mask    = enc_padding_mask,
                ** kwargs
            )
        
        return output

    def call(self,
             inputs,
             training   = False,
             encoder_mask   = None,
             decoder_mask   = None,
             
             return_attention   = None,
             return_states      = None,
             return_mask        = None,
             return_logits      = None,
             
             ** kwargs
            ):
        encoder_inputs, decoder_inputs = inputs[: -2], inputs[-2 :]
        
        encoder_out, encoder_attn, encoder_states, encoder_mask, types = self.encode(
            encoder_inputs, mask = encoder_mask, training = training, 
            return_attention = True, return_mask = True, return_states = True, ** kwargs
        )
        
        output, logits, decoder_attn, decoder_states, decoder_mask = self.decode(
            encoder_out,
            decoder_inputs  = decoder_inputs,
            encoder_out_types   = types,
            
            training    = training,
            decoder_mask    = decoder_mask,
            enc_padding_mask    = encoder_mask,
            
            return_attention    = True,
            return_states       = True,
            return_mask         = True,
            return_logits       = True,
            
            ** kwargs
        )
        
        return self.decoder.decoder.format_output(
            output, logits  = logits,                            return_logits    = return_logits,
            attn_weights    = (encoder_attn, decoder_attn),     return_attention = return_attention,
            states          = (encoder_states, decoder_states), return_states    = return_states,
            mask            = (encoder_mask, decoder_mask),     return_mask      = return_mask
        )

    def infer(self,
              inputs,
              training = False,
              encoder_mask = None, 
                
              return_attention   = None,
              return_states      = None,
              return_mask        = None,
              return_logits      = None,
              ** kwargs
             ):
        encoder_out, encoder_attn, encoder_states, encoder_mask, types = self.encode(
            inputs, mask = encoder_mask, training = training, 
            return_attention = True, return_mask = True, return_states = True, ** kwargs
        )
        
        decoder_out, decoder_attn = self.decode(
            encoder_out,
            decoder_inputs  = None,
            encoder_out_types   = types,
            
            training    = training,
            enc_padding_mask    = encoder_mask,
            
            return_attention    = True,
            return_states       = True,
            return_mask         = True,
            return_logits       = True,
            
            ** kwargs
        )
        
        return self.decoder.decoder.format_output(
            output, logits  = logits,                            return_logits    = return_logits,
            attn_weights    = (encoder_attn, decoder_attn),     return_attention = return_attention,
            states          = (encoder_states, decoder_states), return_states    = return_states,
            mask            = (encoder_mask, decoder_mask),     return_mask      = return_mask
        )

custom_objects  = {
    'MAG'   : MAG
}

custom_functions    = custom_objects
