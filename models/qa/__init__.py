
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

from models.qa.mag import MAG
from models.qa.rag import RAG
from models.qa.answer_retriever import AnswerRetriever
from models.qa.answer_generator import AnswerGenerator
from models.qa.question_generator import QuestionGenerator
from models.qa.context_retriever import ContextRetriever
from models.qa.text_encoder_decoder import TextEncoderDecoder
from models.qa.answer_generator_split import AnswerGeneratorSplit

from models.qa.web_utils import search_on_web

_default_pred_config    = {
    'method'            : 'beam',
    'negative_mode'     : 'doc',
    'max_negatives'     : 25,
    'negative_select_mode'  : 'linear',
    'max_input_length'  : 512
}

def _get_model_name(model = None, lang = None):
    if model is not None: return model
    
    global _pretrained
    
    if lang not in _pretrained:
        raise ValueError("Unknown language : {}, no default model set".format(lang))
        
    return _pretrained[lang]

def answer(question, model = None, lang = None, ** kwargs):
    from models import get_pretrained
    
    model_name  = _get_model_name(model = model, lang = lang)
    model   = get_pretrained(model_name)
    
    return model.predict(question, ** kwargs)

def answer_from_web(question, url = None, engine = None, test_all_engines = True,
                    n = 5, site = 'en.wikipedia.org', lang = 'en', ** kwargs):
    import requests
    
    from utils.text import parse_html
    
    if url is None:
        result = search_on_web(
            question, n = n, site = site, engine = engine, test_all_engines = test_all_engines
        )
        url, engine = result['urls'], result['engine']
    if not isinstance(url, (list, tuple)): url = [url]
    
    pages = [requests.get(url_i) for url_i in url]
    pages = [p.content.decode('utf-8') for p in pages if p.status_code == 200]

    parsed = []
    for html in pages: parsed.extend(parse_html(html))
    
    if len(parsed) == 0: parsed = [{'title' : '', 'text' : '<no result>'}]
    
    data = [{
        'question'      : question,
        'paragraphs'    : [p['text'] for p in parsed],
        'titles'        : [p['title'] for p in parsed],
        'engine'        : engine,
        'urls'          : url[0] if len(url) == 1 else url
    }]

    return answer(data, lang = lang, ** {** _default_pred_config, ** kwargs})

_models = {
    'MAG'           : MAG,
    'RAG'           : RAG,
    'QARetriever'   : AnswerRetriever,
    'AnswerGenerator'   : AnswerGenerator,
    'QuestionGenerator' : QuestionGenerator,
    'ContextRetriever'  : ContextRetriever,
    'TextEncoderDecoder'    : TextEncoderDecoder,
    'AnswerGeneratorSplit'  : AnswerGeneratorSplit
}

_pretrained = {
    'en'    : 'm5_nq_coqa_newsqa_mag_off_entq_ct_wt_ib_2_2_dense'
}