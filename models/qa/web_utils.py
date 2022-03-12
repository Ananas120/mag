
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

import json
import urllib
import logging
import requests

_default_engine = 'google'

_ddg_api_url    = 'http://api.duckduckgo.com/'
_bing_api_url   = 'http://www.bing.com/search'

def search_on_web(query, * args, engine = None, test_all_engines = True, ** kwargs):
    global _default_engine
    
    if engine is None: engine = _default_engine

    if engine not in _search_engines:
        raise ValueError('Unknown search engine !\n  Accepted : {}\n  Got : {}'.format(
            tuple(_search_engines.keys()), engine
        ))
    
    if test_all_engines:
        engines  = [engine] + [e for e in _search_engines.keys() if e != engine]
    else:
        engines = [engine]
    for engine_i in engines:
        try:
            logging.info('Try query {} on engine {}...'.format(query, engine_i))
            urls    = _search_engines[engine_i](query, * args, ** kwargs)
            if len(urls) == 0:
                logging.warning('No result with engine {} for query {}, trying another search engine !'.format(engine_i, query))
                continue
            
            result = {'engine' : engine_i, 'urls' : urls}
            
            if _default_engine is None: _default_engine = engine_i
            
            return result
        except Exception as e:
            logging.error('Error with engine {} : {}, trying next engine'.format(engine_i, str(e)))
            if _default_engine == engine: _default_engine = None
    
    return {'engine' : None, 'urls' : []}

def search_on_google(query, n = 10, site = None, ** kwargs):
    """ Return a list of url's for a given query """
    import googlesearch as google
    
    if site is not None: query = '{} site:{}'.format(query, site)
    
    results = []
    for res in google.search(query, safe = 'on', ** kwargs):
        results.append(res)
        if len(results) == n: break
    
    return results

def search_on_ddg(query, n = 10, site = None, ** kwargs):
    if site is not None: query = '{} site:{}'.format(query, site)

    params = {
        'q'     : query,
        'o'     : 'json',
        'kp'    : '1',
        'no_redirect'   : '1',
        'no_html'       : '1'
    }

    url = '{}?{}'.format(_ddg_api_url, urllib.parse.urlencode(params))
    res = requests.get(url, headers = {'User-Agent' : 'mag'})
    
    if len(res.content) == 0 or not res.json()['AbstractURL']: return []
    return res.json()['AbstractURL']

def search_on_bing(query, n = 10, site = None, ** kwargs):
    from bs4 import BeautifulSoup
    
    if site is not None: query = query + ' site:' + site
    params = {
        'q'     : '+'.join(query.split())
    }
    encoded = '&'.join(['{}={}'.format(k, v) for k, v in params.items()])
    url = '{}?{}'.format(_bing_api_url, encoded)
    res = BeautifulSoup(requests.get(url, headers = {'User-Agent' : 'mag'}).text)

    raw_results = res.find_all('li', attrs= {'class' : 'b_algo'})
    links = []
    for raw in raw_results:
        link = raw.find('a').get('href')
        if link: links.append(link)
        
    return links[:n]

_search_engines = {
    'google'    : search_on_google,
    'bing'      : search_on_bing,
    'ddg'       : search_on_ddg
}
