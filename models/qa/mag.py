import tensorflow as tf

import custom_architectures.transformers_arch.mag_arch as mag_arch

from loggers import timer
from models.qa.base_qa_generator import BaseQAGenerator
from models.qa.answer_generator_split import AnswerGeneratorSplit

class MAG(AnswerGeneratorSplit):
    def _build_model(self, pretrained, ** kwargs):
        super(BaseQAGenerator, self)._build_model(
            model = mag_arch.MAG.from_pretrained(
                pretrained, return_attention = False, ** kwargs
            )
        )

    @property
    def subsampling_factor(self):
        return self.model.hparams.encoder_subsampling_step

    def __str__(self):
        des = super().__str__()
        des += "- # of embedding layers : {}\n".format(len(self.encoder.embedding_layers))
        des += "- # of memory layers : {}\n".format(len(self.encoder.memory_layers))
        des += "- Subsampling factor : {}\n".format(self.subsampling_factor)
        des += "- Subsampling mode : {}\n".format(self.model.hparams.encoder_subsampling_mode)
        return des
    
    @timer(log_if_root = False)
    def encode(self, inputs, training = False, merge_contexts = False, ** kwargs):
        q_not_subsampling = False if self.subsample_question else ([True] + [False] * (len(inputs) // 2 - 1))

        embeddings, mask, types = self.model.encode(
            inputs, training    = training,
            positional_offset   = self.context_offset,
            merge_contexts      = merge_contexts or self.force_merging or (training and self.in_batch_negatives),
            force_not_subsampling   = q_not_subsampling,
            return_attention    = False,
            return_states   = False,
            return_mask     = True,
            ** kwargs
        )
        
        return (embeddings, mask, types)
    
    @timer(name = 'prediction', log_if_root = False)
    def call(self, inputs, training = False, ** kwargs):
        encoder_outputs, enc_padding_mask, enc_types = self.encode(inputs[:-2], training = training, ** kwargs)
        
        return self.model.decode(
            encoder_outputs,
            decoder_inputs  = inputs[-2 :],
            encoder_out_types   = enc_types,
            
            training    = training,
            enc_padding_mask    = enc_padding_mask,
            return_attention    = False
        )
    
    @timer(name = 'inference', log_if_root = False)
    def infer(self, inputs, training = False, ** kwargs):
        kwargs.setdefault('max_length', self.max_output_length)
        encoder_outputs, enc_padding_mask, enc_types = self.encode(
            inputs, training = training, ** kwargs
        )
        
        return self.model.decode(
            encoder_outputs,
            decoder_inputs  = None,
            encoder_out_types   = enc_types,
            
            training    = training,
            enc_padding_mask    = enc_padding_mask,
            return_attention    = False,
            return_logits       = False,
            return_mask     = False,
            ** kwargs
        )
