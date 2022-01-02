import tensorflow as tf

from models.qa.base_qa import BaseQAModel, find_index
from custom_architectures.transformers_arch.bert_arch import BertQA

class AnswerRetriever(BaseQAModel):
    def __init__(self, * args, pretrained = 'bert-base-uncased',** kwargs):
        super().__init__(* args, pretrained = pretrained, ** kwargs)
    
    def _build_model(self, pretrained, ** kwargs):
        super()._build_model(
            model = BertQA.from_pretrained(pretrained, return_attention = False, ** kwargs)
        )
    
    @property
    def input_signature(self):
        return (
            tf.TensorSpec(shape = (None, None), dtype = tf.int32),  # text (tokens ids)
            tf.TensorSpec(shape = (None,), dtype = tf.int32),       # text length
            tf.TensorSpec(shape = (None, None), dtype = tf.int32)   # token type (0 = question, 1 = context)
        )
    
    @property
    def output_signature(self):
        return (
            tf.TensorSpec(shape = (None, ), dtype = tf.int32),  # Start idx
            tf.TensorSpec(shape = (None, ), dtype = tf.int32)   # End idx
        )
        
    def compile(self, loss = 'QARetrieverLoss', ** kwargs):
        super().compile(loss = loss, ** kwargs)
    
    def get_input(self, data):
        encoded_text, token_types = self.tf_format_input(data)
        
        return (encoded_text, len(encoded_text), token_types)

    def get_output(self, data, inputs = None):
        if inputs is None: inputs = self.get_input(data)
        encoded_text = inputs[0]
        
        encoded_answer = tf.py_function(
            self.encode_text, [data['answers'], False], Tout = tf.int32
        )
        encoded_answer.set_shape([None])

        start_idx = find_index(encoded_text, encoded_answer)
        
        return start_idx, start_idx + len(encoded_answer)

    def get_dataset_config(self, ** kwargs):
        kwargs.update({
            'batch_before_map'  : True,
            'padded_batch'      : True,
            'pad_kwargs'        : {
                'padded_shapes'     : (
                    ((None,), (), (None,)), ((), ())
                ),
                'padding_values'    : ((self.blank_token_idx, 0, 0), (0, 0))
            }
        })
        
        return super().get_dataset_config(** kwargs)
