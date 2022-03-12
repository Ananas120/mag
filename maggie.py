
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
import discord

from threading import Lock

from loggers import set_level
from utils import load_json, dump_json
from models.qa import MAG, _pretrained, answer_from_web

set_level('info')

MAX_LENGTH  = 150

_emojis_scores  = {
    '1'     : '1️⃣',
    '2'     : '2️⃣',
    '3'     : '3️⃣',
    '4'     : '4️⃣',
    '5'     : '5️⃣'
}
_correct_emoji  = {
    'true'  : '✅',
    'false' : '❎'
}
_emojis  = _emojis_scores

_emoji_to_value  = {v : k for k, v in _emojis.items()}

_correctness_key    = 'correctness'
_evaluations    = {
    _correctness_key    : None,
    'quality'   : '**Quality** for Q&A {} \n1 = poor\n3 = correct but not detailed\n5 = well detailed'
}

URL_GIT         = 'https://github.com/Ananas120/mag'
URL_REPORT      = 'https://www.overleaf.com/read/cccygzfthhrk'
HELP_MESSAGE    = """
**MAGgie** is a Q&A bot created for the @Mew Master thesis (which is about Q&A models in Deep Learning).

The objective is to test the proposed approach in real-world conditions : try to ask questions and please evaluate the answers by adding reactions !

Commands :
    .help       : get information on commands
    .git        : get the github project's URL
    .report     : get the master thesis' report URL (overleaf)
    .results    : show statistics on reactions
    .ask <question>     : ask a question to the bot

Thank you for your help ! Your evaluation is crucial to show the performance of the technique !
"""

def get_results(directory):
    results = {}
    n_react = 0
    for q_file in os.listdir(directory):
        data = load_json(os.path.join(directory, q_file))

        has_react = False
        for key, _ in _evaluations.items():
            results.setdefault(key, {})
            
            majority = sorted(data.get(key, {}).items(), key = lambda e: len(e[1]), reverse = True)
            if len(majority) > 0 and len(majority[0][1]) > 0:
                has_react = True
                score     = majority[0][0]
                results[key].setdefault(score, 0)
                results[key][score] += 1
        
        if has_react: n_react += 1

    return results, n_react

class Maggie(discord.Client):
    def __init__(self, directory = 'maggie', ** kwargs):
        super().__init__(command_prefix = '!', ** kwargs)
        
        self.directory  = directory
        
        self.mutex  = Lock()
        self.mutex_react    = Lock()
        
        os.makedirs(self.responses_directory, exist_ok = True)
        
        _ = MAG(nom = _pretrained['en'])
    
    @property
    def responses_directory(self):
        return os.path.join(self.directory, 'responses')
    
    @property
    def user_name(self):
        return self.user
    
    @property
    def user_id(self):
        return self.user.id
    
    async def on_ready(self):
        logging.info("{} with ID {} started !".format(self.user_name, self.user_id))
    
    async def on_message(self, context):
        if not (len(context.content) > 0 and context.content[0] in ('.', '!')): return
        
        infos = context.content[1:].split()
        command, msg = infos[0], ' '.join(infos[1:])
        
        logging.info('Get command {}'.format(command))
        
        if not hasattr(self, command):
            await ctx.channel.send('Unknown command : {}\nUse .help for help ;)'.format(command))
            return
        
        await getattr(self, command)(msg, context = context)
    
    async def on_raw_reaction_add(self, reaction):
        if reaction.member.id == self.user_id: return
        
        q_id, eval_type, score = await self.get_infos_from_reaction(reaction)
        if q_id is None: return

        with self.mutex_react:
            filename    = self.get_filename(q_id)
            
            data = load_json(filename)
            
            data.setdefault(eval_type, {})
            if reaction.member.id not in data[eval_type].get(score, []):
                logging.info('User {} adds score {} for type {} on question ID {} !'.format(
                    reaction.member.id, score, eval_type, q_id
                ))
                data[eval_type].setdefault(score, []).append(reaction.member.id)
                
                dump_json(filename, data, indent = 4)
                

    async def on_raw_reaction_remove(self, reaction):
        if reaction.member is None or reaction.member.id == self.user_id: return
        
        q_id, eval_type, score = await self.get_infos_from_reaction(reaction)
        if q_id is None: return
        
        with self.mutex_react:
            filename    = self.get_filename(q_id)
            
            data = load_json(filename)
            
            data.setdefault(eval_type, {})
            if reaction.member.id in data[eval_type].get(score, []):
                logging.info('User {} removes score {} for type {} on question ID {} !'.format(
                    reaction.member.id, score, eval_type, q_id
                ))
                data[eval_type][score].remove(reaction.member.id)
                
                dump_json(filename, data, indent = 4)

    async def help(self, msg, context):
        await context.channel.send(HELP_MESSAGE)
        
    async def hello(self, msg, context):
        await context.channel.send('Hello {} !'.format(context.author.name))
    
    async def git(self, msg, context):
        await context.channel.send(
            'This project is open-source at {} ! :smile:'.format(URL_GIT)
        )
        
    async def report(self, msg, context):
        await context.channel.send(
            'The master thesis\' report is avaialble at {} ! :smile:'.format(URL_REPORT)
        )

        
    async def ask(self, question, context):
        with self.mutex:
            logging.info('Question {} from user {} !'.format(question, context.author))
            answer  = answer_from_web(question)[0]
            answer['question']  = question
        
        result = await context.channel.send(self.format_answer(question, answer))
        q_id = result.id
        logging.info("Question ID : {}".format(q_id))
        
        for eval_type, msg in _evaluations.items():
            answer.setdefault(eval_type, {})
            for score in _emojis_scores.keys():
                answer[eval_type].setdefault(score, [])
        
        dump_json(self.get_filename(q_id), answer, indent = 4)
        
        
        await self.add_default_emojis(result)
        
        for eval_type, msg in _evaluations.items():
            if msg is None: continue
            
            result = await context.channel.send(msg.format(q_id))
            await self.add_default_emojis(result)

    
    async def results(self, msg, context):
        res, n_react = get_results(self.responses_directory)
        n = len(os.listdir(self.responses_directory))
        
        des = '# questions : {} ({} with reactions)'.format(n, n_react)
        for k, v in res.items():
            print(k, v)
            des += '\n{} :'.format(k.capitalize())
            for ki, vi in sorted(v.items(), key = lambda p: p[0]):
                des += '\n- {} : {} ({:.2f} %)'.format(
                    ki, vi, int(vi * 100 / n_react) if n_react > 0 else 0
                )
        await context.channel.send(des)
        
    async def add_default_emojis(self, ctx):
        for k, emoji in _emojis.items():
            res = await ctx.add_reaction(emoji)
    
    async def get_infos_from_reaction(self, reaction):
        msg_id = reaction.message_id
        if os.path.exists(self.get_filename(msg_id)):
            q_id        = msg_id
            eval_type   = _correctness_key
        else:
            channel = self.get_channel(reaction.channel_id)
            message = await channel.fetch_message(reaction.message_id)
            
            if 'Q&A' not in message.content:
                return None, None, None
                
            q_id    = message.content.split('\n')[0].split()[-1]
            eval_type   = message.content.split()[0].lower().replace('*', '')
            
        return q_id, eval_type, _emoji_to_value.get(reaction.emoji.name, reaction.emoji.name)
    
    def get_filename(self, q_id):
        return os.path.join(self.responses_directory, '{}.json'.format(q_id))
    
    def format_answer(self, question, answer, add_url = False, n_passages = 1):
        des = "Question : **{}**\n".format(question)
        if len(answer.get('urls', [])) > 0 and add_url:
            des += "Page(s) used : {}\n".format(answer['urls'])
        
        passages = {}
        
        des += "\nCandidates :\n"
        for i, cand in enumerate(answer['candidates']):
            des += "- **Candidate #{} (score : {:.3f}) : {}**\n".format(
                i + 1, cand['score'], cand['text']
            )
            
            if len(cand.get('passages', [])) > 0:
                des += "  Passages :\n"
                for j, sent in enumerate(cand.get('passages', [])[:n_passages]):
                    if sent in passages: continue
                    passages[sent] = True
                    des += "  -  Passage #{} : {}\n".format(
                        j, sent if len(sent) < MAX_LENGTH else (sent[:MAX_LENGTH] + ' [...]')
                    )
            des += "\n"
        
        des += "\n**Please add reactions for correct answers** and reactions for the **quality** (see next message) ! :smile:"
        
        return des

if __name__ == '__main__':
    token = os.environ.get('DISCORD_BOT_TOKEN', None)
    if token is None:
        raise ValueError('You should give the discord bot token as `DISCORD_BOT_TOKEN` env variable !')

    intents = discord.Intents.default()

    bot = Maggie(directory = 'memoire_results/maggie', intents = intents)
    bot.run(token)