
# Copyright (C) 2022 Langlois Quentin. All rights reserved.
# Licenced under the Affero GPL v3 Licence (the "Licence").
# you may not use this file except in compliance with the License.
# See the "LICENCE" file at the root of the directory for the licence information.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import logging
import numpy as np
import pandas as pd
import tensorflow as tf

from loggers import timer
from utils import load_json, dump_json, pad_batch
from utils.text import extract_sentence
from models.base_model import _compile_fn
from models.qa.base_qa import BaseQAModel
from custom_architectures.transformers_arch.bart_arch import Bart
from custom_architectures.transformers_arch.gpt2_arch import GPT2

time_logger = logging.getLogger('timer')

_pred_classic_infos = [
    'question', 'context', 'title', 'paragraphs', 'titles', 'answers'
]

def infer_to_str(text, score, indent = 0):
    _indentation = ' ' * indent
    if not isinstance(text, (list, tuple)):
        return '{}Inference ({:.3f}) : {}'.format(_indentation, score, text)
    des = '{}Inference :'.format(_indentation)
    for j, (s, txt) in enumerate(zip(score, text)):
        des += '\n{}  #{} ({:.3f}) : {}'.format(_indentation, j, s, txt)
    return des

class BaseGenerator(BaseQAModel):
    def __init__(self,
                 * args,
                 max_output_length  = 1024,
                 pretrained         = 'facebook/bart-large',
                 ** kwargs
                ):
        self.max_output_length = max_output_length
        
        self.show_input = kwargs.get('show_input', True)
        super().__init__(* args, pretrained = pretrained, ** kwargs)
    
    def init_train_config(self,
                          max_output_length     = None,
                          teacher_forcing_eval  = True,
                          eval_infer_config     = {},
                          ** kwargs
                         ):
        if max_output_length: self.max_output_length = max_output_length
        self.teacher_forcing_eval   = teacher_forcing_eval
        self.eval_infer_config      = eval_infer_config
        
        super().init_train_config(** kwargs)

    def _build_model(self, pretrained, ** kwargs):
        kwargs.update({'return_attention' : False, 'return_hidden_states' : False})
        if 'bart' in pretrained:
            model = Bart.from_pretrained(pretrained, ** kwargs)
        elif 'gpt' in pretrained:
            model = GPT2.from_pretrained(pretrained, ** kwargs)
        super()._build_model(model = model)
    
    @property
    def training_hparams(self):
        return super().training_hparams(
            max_output_length = None, teacher_forcing_eval = True, eval_infer_config = {}
        )

    @property
    def is_encoder_decoder(self):
        return hasattr(self.model, 'decoder')
    
    @property
    def token_length_signature(self):
        return (
            tf.TensorSpec(shape = (None, None), dtype = tf.int32),  # tokens
            tf.TensorSpec(shape = (None, ),     dtype = tf.int32)   # length
        )
    
    @property
    def multi_token_length_signature(self):
        return (
            tf.TensorSpec(shape = (None, None, None),   dtype = tf.int32),  # tokens
            tf.TensorSpec(shape = (None, None),         dtype = tf.int32)   # length
        )

    @property
    def input_signature(self):
        signature = self.token_length_signature
        if self.is_encoder_decoder: signature = signature + signature
        return signature
    
    @property
    def output_signature(self):
        signature = self.multi_token_length_signature #token_length_signature
        if self.is_encoder_decoder:
            return signature
        return signature + (tf.TensorSpec(shape = (None, ), dtype = tf.int32), )
    
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
    
    def compile(self,
                loss        = 'TextLoss',
                loss_config = {},
                metrics     = ['TextAccuracy', 'F1'],
                metrics_config  = {},
                optimizer_config    = {'lr' : 1e-5},
                ** kwargs
               ):
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
    
    def encode_multi_answers(self, answers, ** kwargs):
        if isinstance(answers, tf.Tensor): answers = answers.numpy()
        if not isinstance(answers, (list, tuple, np.ndarray)): answers = [answers]
        
        encoded = [
            self.format_output(answer = a if not isinstance(a, bytes) else a.decode('utf-8'))[0]
            for a in answers
        ]
        
        return pad_batch(encoded, pad_value = self.blank_token_idx), [len(a) for a in encoded]

    def tf_format_multi_output(self, data):
        answers = data.get('answers', ['']) if isinstance(data, dict) else data
        
        encoded_outputs, lengths    = tf.py_function(
            self.encode_multi_answers, [answers], Tout = [tf.int32, tf.int32]
        )
        encoded_outputs.set_shape([None, None])
        lengths.set_shape([None])
        
        valid_outputs   = lengths <= self.max_output_length

        encoded_outputs = tf.boolean_mask(encoded_outputs, valid_outputs)
        lengths         = tf.boolean_mask(lengths, valid_outputs)
        
        if len(lengths) > 0:
            encoded_outputs = encoded_outputs[:, : tf.reduce_max(lengths)]
        
        return encoded_outputs, lengths

    def get_output(self, data, inputs = None):
        tokens, lengths = self.tf_format_multi_output(data)
        
        return tokens, lengths

    def encode_data(self, data):
        inputs, outputs = super().encode_data(data)
        
        if not self.is_encoder_decoder and len(inputs) == 2:
            inp_tokens, inp_length = inputs
            out_tokens, out_length = outputs
            
            tokens  = tf.concat([inp_tokens[:-1], out_tokens[1:]], axis = -1)
            
            inputs  = (tokens, inp_length + out_length - 2)
            outputs = (tokens, out_length - 1, inp_length - 1)

        return inputs, outputs
    
    def filter_inputs(self, inputs):
        return tf.shape(inputs[0])[-1] <= self.max_input_length
        
    def filter_outputs(self, outputs):
        return len(outputs[0]) > 0 and tf.shape(outputs[0])[-1] <= self.max_output_length
    
    def filter_data(self, inputs, outputs):
        return self.filter_inputs(inputs) and self.filter_outputs(outputs)
    
    def augment_data(self, inputs, outputs):
        inp_tokens, inp_length = inputs[:2]
        
        inp_tokens, inp_length = self.augment_text(inp_tokens, inp_length)
        
        return (inp_tokens, inp_length) + inputs[2:], outputs
    
    def preprocess_data(self, inputs, outputs):
        if self.is_encoder_decoder:
            answer, answer_length = outputs
            answer_in, answer_in_length = answer[..., :-1], answer_length -1
            
            if len(tf.shape(answer_in)) == 3:
                answer_in, answer_in_length = answer_in[:, 0], answer_in_length[:, 0]
                answer_in = answer_in[:, : tf.reduce_max(answer_in_length)]
            
            return inputs + (answer_in, answer_in_length), (answer[..., 1:], answer_length - 1)
        
        inp_tokens, inp_lengths = inputs
        out_tokens, out_lengths, skip_length = outputs
        return (
            (inp_tokens[:, :-1], inp_lengths - 1),
            (out_tokens[:, 1:], out_lengths, skip_length - 1)
        )
    
    def get_dataset_config(self, ** kwargs):
        inp_signature, out_signature = self.input_signature, self.output_signature
        if self.is_encoder_decoder: inp_signature = inp_signature[:-2]
        
        kwargs.update({
            'batch_before_map'  : True,
            'padded_batch'      : True,
            'pad_kwargs'        : {
                'padded_shapes'     : (
                    tuple([tuple(sign.shape)[1:] for sign in inp_signature]),
                    tuple([tuple(sign.shape)[1:] for sign in out_signature])
                ),
                'padding_values'    : (
                    tuple([self.blank_token_idx, 0] * (len(inp_signature) // 2)),
                    tuple([self.blank_token_idx] + [0] * (len(out_signature) - 1))
                )
            }
        })
        
        return super().get_dataset_config(** kwargs)

    def eval_step(self, batch):
        inputs, target = batch
        
        if self.teacher_forcing_eval:
            y_pred = self(inputs, training = False)
        else:
            if self.is_encoder_decoder: inputs = inputs[:-2]
            y_pred = self.infer(inputs, training = False, ** self.eval_infer_config).tokens

        return self.update_metrics(target, y_pred)

    @timer
    def predict_with_target(self, batch, epoch = None, step = None, prefix = None, 
                            directory = None, n_pred = 5, ** kwargs):
        inputs, output = batch
        inputs  = [inp[:n_pred] for inp in inputs]
        outputs = [out[:n_pred] for out in output]
        answers, answers_length = outputs[:2]
        infer_inputs    = inputs[:-2] if self.is_encoder_decoder else inputs
        
        pred    = self(inputs, training = False, ** kwargs)
        infer   = self.infer(
            infer_inputs, max_length = tf.shape(answers)[-1], early_stopping = False,
            ** self.eval_infer_config, ** kwargs
        )
        
        pred_text   = self.decode_text(pred)
        infer_text  = self.decode_text(infer.tokens)
        
        input_text  = self.decode_text(inputs[0]) if self.show_input else None
        target_text = self.decode_text(answers)
        
        preds = []
        for i in range(len(target_text)):
            preds.append("Prediction {} / {} :\n{}  Target     : {}\n  Prediction : {}\n{}".format(
                i + 1, len(target_text),
                "" if input_text is None else "  Input      : {}\n".format(input_text[i]),
                target_text[i],
                pred_text[i],
                '' if infer_text is None else infer_to_str(infer_text[i], infer.score[i], indent = 2)
            ))
        
        logging.info("\n".join(preds))
    
    @timer
    def predict(self,
                question,
                
                title   = None,
                context = None,
                
                metrics = None,
                
                save        = False,
                overwrite   = False,
                directory   = None,
                filename    = 'map.json',
                
                tqdm    = lambda x: x,
                
                ** kwargs
               ):
        time_logger.start_timer('processing')

        pred_config = self.training_hparams.extract(kwargs, pop = True)
        self.init_train_config(** pred_config)
        
        logging.dev('Predicting config :\n{}'.format(pred_config))

        if not hasattr(self, '_compiled_infer'):
            self._compiled_infer    = _compile_fn(
                self.infer,
                run_eagerly = kwargs.pop('run_eagerly', False)
            )
        
        if metrics is not None: metrics = self.get_compiled_metrics(metrics, add_loss = False)

        if isinstance(question, pd.DataFrame): question = question.to_dict('record')
        if not isinstance(question, list): question = [question]

        if context is not None:
            if not isinstance(context, list) or len(context) != len(question): context = [context]
            if len(context) == 1 and len(question) > 1: context = context * len(question)

            if title is not None:
                if not isinstance(title, list) or len(title) != len(context): title = [title]
                if len(title) == 1 and len(context) > 1: title = title * len(context)
        
        
        data = question if context is None else []
        if context is not None:
            for i, q in enumerate(question):
                if not isinstance(q, dict): q = {'question' : q}
                ctx = context[i] if len(context) == len(question) else context
                
                if not isinstance(ctx, dict):
                    key = 'paragraphs' if isinstance(ctx, (list, tuple)) else 'context'
                    ctx = {key : ctx}
                    if title is not None:
                        key = 'titles' if key == 'paragraphs' else 'title'
                        ctx[key] = title[i] if len(title) == len(question) else title
                
                data.append({** q, ** ctx})

        time_logger.stop_timer('processing')

        infos_pred = {}
        if save:
            if directory is None: directory = self.pred_dir
            if filename is None or '.json' in directory: filename, directory = directory, None
            else: filename = os.path.join(directory, filename)
            if directory is not None: os.makedirs(directory, exist_ok = True)

            infos_pred = load_json(filename)

        answers = []
        for idx, row in enumerate(tqdm(data)):
            question    = row['question']
            context     = row['context'] if 'paragraphs' not in row else row['paragraphs']
            
            ctx_key     = context if not isinstance(context, list) else '\n\n'.join(context)

            if overwrite or not (question in infos_pred and ctx_key in infos_pred[question]):
                inputs = [tf.expand_dims(inp, axis = 0) for inp in self.get_input(row)]

                if not self.filter_inputs([inp[0] for inp in inputs]):
                    logging.warning('Too long data at index {} : {}'.format(
                        idx, [tuple(inp.shape) for inp in inputs]
                    ))
                    continue
                
                additional_infos    = {
                    k : v for k, v in row.items() if k not in _pred_classic_infos
                }
                
                pred = self._compiled_infer(inputs, training = False, ** kwargs)

                scores      = pred.score[0].numpy()
                pred_text   = self.decode_text(pred.tokens, remove_tokens = True)[0]
                if not isinstance(pred_text, (list, tuple)):
                    pred_text, scores = [pred_text], [scores]

                infos_pred.setdefault(question, {})
                infos_pred[question][ctx_key] = {
                    ** additional_infos, 'candidates' : []
                }
                
                target = []
                if 'answers' in row and metrics is not None:
                    infos_pred[question][ctx_key]['target'] = row['answers']
                    target = [
                        tf.expand_dims(out, axis = 0) for out in self.get_output(row)
                        if len(out) > 0
                    ]

                for i, (txt, s) in enumerate(zip(pred_text, scores)):
                    metrics_i = {}
                    if len(target) > 0:
                        time_logger.start_timer('metrics')
                        
                        metrics.reset_states()
                        metrics.update_state(target, pred.tokens[:, i])
                        metrics_i = {
                            name : val for name, val in zip(metrics.metric_names, metrics.result().numpy())
                        }
                        
                        time_logger.stop_timer('metrics')
                    
                    passages    = []
                    if '.' in txt:
                        if not isinstance(context, (list, tuple)): context = [context]
                        passages    = [c for c in context if txt in c]
                    else:
                        passages    = extract_sentence(ctx_key, txt)
                        
                    infos_i = {
                        'text'  : txt,
                        'score' : s,
                        'passages'  : passages,
                        ** metrics_i
                    }
                    infos_pred[question][ctx_key]['candidates'].append(infos_i)

            answers.append(infos_pred[question][ctx_key])

        if save:
            dump_json(filename, infos_pred, indent = 4)

        return answers
        
    def get_config(self, * args, ** kwargs):
        config = super().get_config(* args, ** kwargs)
        config['max_output_length'] = self.max_output_length
        return config