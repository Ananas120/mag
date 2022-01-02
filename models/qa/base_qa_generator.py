import logging
import tensorflow as tf

from loggers import timer
from models.qa.base_qa import BaseQAModel
from custom_architectures.transformers_arch.bart_arch import Bart

time_logger = logging.getLogger('timer')

class BaseQAGenerator(BaseQAModel):
    def __init__(self, * args, max_output_length = 1024, pretrained = 'facebook/bart-large', ** kwargs):
        self.max_output_length = max_output_length
        
        self.show_input = kwargs.get('show_input', True)
        super().__init__(* args, pretrained = pretrained, ** kwargs)
    
    def init_train_config(self, max_output_length = None, teacher_forcing_eval = True, ** kwargs):
        if max_output_length: self.max_output_length = max_output_length
        self.teacher_forcing_eval = teacher_forcing_eval
        
        super().init_train_config(** kwargs)

    def _build_model(self, pretrained, ** kwargs):
        super()._build_model(
            model = Bart.from_pretrained(pretrained, return_attention = False, ** kwargs)
        )
    
    @property
    def training_hparams(self):
        return super().training_hparams(max_output_length = None, teacher_forcing_eval = True)

    @property
    def input_signature(self):
        return (
            tf.TensorSpec(shape = (None, None), dtype = tf.int32),  # text (tokens ids) for encoder input
            tf.TensorSpec(shape = (None,), dtype = tf.int32),       # text length for encoder input
            tf.TensorSpec(shape = (None, None), dtype = tf.int32),  # text (tokens ids) for decoder input
            tf.TensorSpec(shape = (None,), dtype = tf.int32)        # text length for decoder input
        )
    
    @property
    def output_signature(self):
        return (
            tf.TensorSpec(shape = (None, None), dtype = tf.int32),  # text (tokens ids)
            tf.TensorSpec(shape = (None,), dtype = tf.int32)        # text length
        )
    
    @property
    def default_metrics_config(self):
        return {
            'pad_value' : self.blank_token_idx,
            'eos_value' : self.eos_token_idx,
            'decode_fn' : lambda text: self.decode_text(text, remove_tokens = True)
        }

    @timer(name = 'inference', log_if_root = False)
    def infer(self, text, text_length = None, ** kwargs):
        if isinstance(text, (list, tuple)): text, text_length = text
        if len(tf.shape(text)) == 1: text = tf.expand_dims(text, axis = 0)
        if text_length is None: text_length = tf.fill([tf.shape(text)[0]], tf.shape(text)[1])
        elif len(tf.shape(text_length)) == 0: text_length = tf.expand_dims(text_length, axis = 0)
        
        kwargs.setdefault('max_length', self.max_output_length)
        return self.model.infer([text, text_length], ** kwargs)
    
    def compile(self, loss = 'TextLoss', loss_config = {}, metrics = ['TextAccuracy', 'F1'],
                metrics_config = {}, optimizer_config = {'lr' : 1e-5}, ** kwargs):
        loss_config['pad_value']    = self.blank_token_idx
        metrics_config.update(self.default_metrics_config)
        
        super().compile(
            loss    = loss,
            metrics = metrics,
            loss_config = loss_config,
            metrics_config  = metrics_config,
            optimizer_config    = optimizer_config,
            ** kwargs
        )
    
    def filter_data(self, inputs, outputs):
        return inputs[1] <= self.max_input_length and outputs[1] <= self.max_output_length
    
    def augment_data(self, inputs, outputs):
        inp_tokens, inp_length = inputs[:2]
        
        inp_tokens, inp_length = self.augment_text(inp_tokens, inp_length)
        
        return (inp_tokens, inp_length) + inputs[2:], outputs
    
    def preprocess_data(self, inputs, outputs):
        answer, answer_length = outputs
        
        return inputs + (answer[:, :-1], answer_length - 1), (answer[:, 1:], answer_length - 1)
    
    def get_dataset_config(self, ** kwargs):
        kwargs.update({
            'batch_before_map'  : True,
            'padded_batch'      : True,
            'pad_kwargs'        : {
                'padded_shapes'     : (
                    ((None,), ()), ((None, ), ())
                ),
                'padding_values'    : ((self.blank_token_idx, 0), (self.blank_token_idx, 0))
            }
        })
        
        return super().get_dataset_config(** kwargs)

    def eval_step(self, batch):
        inputs, target = batch
        
        if self.teacher_forcing_eval:
            y_pred = self(inputs, training = False)
        else:
            y_pred = self.infer(inputs[:-2], training = False)

        return self.update_metrics(target, y_pred)

    def predict_with_target(self, batch, epoch = None, step = None, prefix = None, 
                            directory = None, n_pred = 5, ** kwargs):
        inputs, output = batch
        inputs = [inp[:n_pred] for inp in inputs]
        answers, answers_length = [out[:n_pred] for out in output]
        
        pred    = self(inputs, training = False)
        infer   = self.infer(
            inputs[:-2], max_length = tf.shape(answers)[1], early_stopping = False, ** kwargs
        )
        
        pred_text   = self.decode_text(pred)
        infer_text  = self.decode_text(infer)
        
        input_text  = self.decode_text(inputs[0]) if self.show_input else None
        target_text = self.decode_text(answers)
        
        preds = []
        for i in range(len(target_text)):
            preds.append("Prediction {} / {} :\n{}  Target     : {}\n  Prediction : {}\n{}".format(
                i + 1, len(target_text),
                "" if input_text is None else "  Input      : {}\n".format(input_text[i]),
                target_text[i],
                pred_text[i],
                "" if infer_text is None else "  Inference  : {}".format(infer_text[i])
            ))
        
        logging.info("\n".join(preds))
    
    @timer
    def predict(self, question, context = None, ** kwargs):
        time_logger.start_timer('processing')
        
        if not isinstance(question, list): question = [question]
        if context is not None:
            if not isinstance(context, list) or len(context) != len(question): context = [context]
        if len(context) == 1 and len(question) > 1: context = context * len(question)

        data = question if context is None else []
        if context is not None:
            for i, q in enumerate(question):
                if not isinstance(q, dict): q = {'question' : q}
                if len(context) == len(question):
                    c = context[i] if isinstance(context[i], dict) else {'context' : context[i]}
                else:
                    c = {'context' : context} if not isinstance(context, dict) else context
                data.append({** q, ** c})

        time_logger.stop_timer('processing')
        
        answers = []
        for row in data:
            inputs = [tf.expand_dims(inp, axis = 0) for inp in self.get_input(row)]

            pred = self.infer(inputs, training = False, ** kwargs)

            answers.append(self.decode_text(pred, remove_tokens = True)[0])

        return answers
        
    def get_config(self, * args, ** kwargs):
        config = super().get_config(* args, ** kwargs)
        config['max_output_length'] = self.max_output_length
        return config