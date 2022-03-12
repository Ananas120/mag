
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

import numpy as np
import pandas as pd
import tensorflow as tf

import custom_architectures.transformers_arch.mag_arch as mag_arch

from loggers import timer
from utils.distance import knn
from models.model_utils import get_model_config
from models.qa.base_generator import BaseGenerator
from models.qa.context_retriever import ContextRetriever

class RAG(BaseGenerator):
    def __init__(self,
                 context_retriever_name,
                 
                 k  = 5,
                 lang   = None,
                 distance_metric    = None,
                 
                 input_format   = ['{question}', '{title}{sep_token}{context}'],
                 
                 ** kwargs
                ):
        if lang is None or distance_metric is None:
            _config = get_model_config(context_retriever_name)
            lang, distance_metric = _config['lang'], _config['distance_metric']

        self.k  = k
        self.distance_metric    = distance_metric
        self.context_retriever_name = context_retriever_name
        
        self.__titles   = None
        self.__contexts = None
        self.__embeddings   = None
        self.__context_retriever    = None
        
        if not isinstance(input_format, (list, tuple)): input_format = [input_format]
        _extended_format = []
        for i, formatted_inp in enumerate(input_format):
            if '{title}' in formatted_inp or '{context}' in formatted_inp:
                _extended_format.extend([
                    formatted_inp.replace('title', 'title_{}'.format(i)).replace('context', 'context_{}'.format(i))
                    for i in range(self.k)
                ])
            else:
                _extended_format.append(formatted_inp)
        
        kwargs.setdefault('show_input', False)
        super().__init__(lang, input_format = _extended_format, ** kwargs)
    
    @property
    def context_retriever(self):
        if self.__context_retriever is None:
            self.__context_retriever = ContextRetriever(nom = self.context_retriever_name)
        return self.__context_retriever
    
    @property
    def titles(self):
        if self.__titles is None:
            raise NotImplementedError()
        return self.__titles
    
    @property
    def contexts(self):
        if self.__contexts is None:
            raise NotImplementedError()
        return self.__contexts
    
    @property
    def embeddings(self):
        if self.__embeddings is None:
            raise NotImplementedError()
        return self.__embeddings
    
    @timer(name = 'inference', log_if_root = False)
    def infer(self,
              inputs    = None,
              question  = None,
              contexts  = None,
              question_embedding    = None,
              context_embeddings    = None,
              ** kwargs
             ):
        if inputs is None:
            raise NotImplementedError()
        return self.model.infer(inputs, ** kwargs)
    
    def _normalize_contexts(self, contexts, titles = None, embeddings = None, ** kwargs):
        assert contexts is not None
        
        if embeddings is None and not isinstance(contexts, pd.DataFrame):
            contexts = self.context_retriever.embed_context(contexts, title = titles)
        
        if isinstance(contexts, pd.DataFrame):
            if 'context_embedding' not in contexts.columns:
                contexts = self.context_retriever.embed_context(contexts)
            
            titles, _contexts, embeddings = [], [], []
            for _, data in contexts.groupby('context_id'):
                item0 = data.iloc[0]
                titles.append(item0['title'])
                _contexts.append(item0['context'])
                embeddings.append(item0['context_embedding'])
            titles, contexts, embeddings = np.array(titles), np.array(_contexts), np.array(embeddings)
        
        if titles is None: titles = np.array([''] * len(contexts))
        return titles, contexts, embeddings
        
    def set_embeddings(self, contexts, titles = None, embeddings = None, overwrite = True, ** kwargs):
        titles, contexts, embeddings = self._normalize_contexts(contexts, titles = titles, embeddings = embeddings)
        
        if overwrite:
            self.__titles   = titles
            self.__contexts = contexts
            self.__embeddings   = embeddings
        else:
            np.append(self.__titles, titles)
            np.append(self.__contexts, contexts)
            np.append(self.__embeddings, embeddings)
    
    def retrieve_k_contexts(self,
                            question    = None,
                            contexts    = None,
                            titles      = None,
                            question_embedding  = None,
                            context_embeddings  = None,
                            k   = None,
                            ** kwargs
                            ):
        if k is None: k = self.k
        if contexts is None and context_embeddings is None:
            titles, contexts, context_embeddings = self.titles, self.contexts, self.embeddings
        else:
            titles, contexts, context_embeddings = self._normalize_contexts(
                contexts, titles = titles, embeddings = context_embeddings
            )

        assert contexts is not None and context_embeddings is not None

        if question_embedding is None:
            question_embedding = self.context_retriever.embed_single_question(question)

        indexes = knn(
            question_embedding,
            context_embeddings,
            ids = None,
            k   = k,
            distance_metric = self.distance_metric,
            max_matrix_size = -1,
            return_index    = True
        )

        return tf.gather(titles, indexes)[0], tf.gather(contexts, indexes)[0]
    
    def format_input(self, question = None, contexts = None, titles = None, answer = None, ** kwargs):
        """ Return formatted inputs tokens (after retrieving k-bests contexts) """
        if titles is None: titles = [''] * self.k
        if isinstance(titles, tf.Tensor):
            if len(tf.shape(titles)) == 0:
                titles = titles.numpy().decode('utf-8')
            else:
                titles = [t.decode('utf-8') for t in titles.numpy()]
        elif not isinstance(titles, (list, tuple, np.ndarray)): titles = [titles] * len(contexts)
        
        kwargs.update({
            'context_{}'.format(i) : contexts[i] if i < len(contexts) else '' for i in range(self.k)
        })
        kwargs.update({
            'title_{}'.format(i) : titles[i] if i < len(titles) else '' for i in range(self.k)
        })
        return self.text_encoder.format(
            self.input_format, question = question, answer = answer, ** kwargs
        )
    
    def get_contexts(self, data):
        """ Return k-best contexts texts (string representation) """
        if 'context_0' in data:
            titles, contexts = [], []
            for i in range(self.k):
                titles.append(data.get('title_{}'.format(i), ''))
                contexts.append(data.get('context_{}'.format(i), ''))
        else:
            titles, contexts = self.retrieve_k_contexts(** data)
        
        return titles, contexts
    
    def get_input(self, data):
        """ Return formatted inputs tokens (after retrieving k-bests contexts) """
        titles, contexts = self.get_contexts(data)
        
        tokens, tokens_type = self.tf_format_input({
            'question' : data['question'], 'context' : contexts, 'title' : titles,
            'answers' : data.get('answers', '')
        })
        
        return tokens, len(tokens)

    def get_config(self, * args, ** kwargs):
        config = super().get_config(* args, ** kwargs)
        config['k'] = self.k
        config['distance_metric']   = self.distance_metric
        config['context_retriever_name']    = self.context_retriever_name
        return config
